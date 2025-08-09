from __future__ import annotations

import asyncio
import inspect
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, ConfigDict

from .tool import LoadedTool


# Compatibility shim: some langchain-core versions mark BaseTool._run as abstract,
# which prevents instantiating async-only tools used in our tests. Relax this.
try:  # pragma: no cover - environment/version dependent
    _run_attr = getattr(BaseTool, "_run", None)
    if _run_attr is not None and getattr(_run_attr, "__isabstractmethod__", False):
        try:
            _run_attr.__isabstractmethod__ = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            abstract_methods = getattr(BaseTool, "__abstractmethods__", frozenset())
            if isinstance(abstract_methods, frozenset) and "_run" in abstract_methods:
                BaseTool.__abstractmethods__ = frozenset(
                    m for m in abstract_methods if m != "_run"
                )
        except Exception:
            pass
except Exception:  # pragma: no cover - defensive guard
    pass

def _run_coro_in_new_loop(coro):
    """Run an async coroutine to completion in a dedicated new event loop thread.

    This avoids calling asyncio.run() when a loop is already running in the current thread.
    """
    import threading

    result_container: dict[str, Any] = {}
    error_container: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            import asyncio as _asyncio

            loop = _asyncio.new_event_loop()
            try:
                _asyncio.set_event_loop(loop)
                result_container["value"] = loop.run_until_complete(coro)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:  # pragma: no cover - rare on supported runtimes
                    pass
                loop.close()
        except BaseException as exc:  # pragma: no cover - hard to trigger deterministically in tests
            error_container["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_container:
        raise error_container["error"]
    return result_container.get("value")


class _SyncProxyTool(BaseTool):
    """Sync proxy that preserves inner schema and handles async-only execution.

    - Keeps args_schema aligned with the inner tool
    - Filters tool_call_schema to avoid fields not present on args_schema
    - Falls back to calling inner._arun with a default config if needed
    """

    inner: BaseTool
    args_schema: type[BaseModel] = BaseModel
    name: str = ""
    description: str = ""

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        # Fill in defaults from inner tool if not provided
        if not getattr(self, "name", None):
            try:
                object.__setattr__(self, "name", getattr(self.inner, "name", "tool"))
            except Exception:
                object.__setattr__(self, "name", "tool")
        if not getattr(self, "description", None):
            try:
                desc = getattr(self.inner, "description", "") or ""
                object.__setattr__(self, "description", desc)
            except Exception:
                object.__setattr__(self, "description", "")

    def _run(self, **kwargs: Any) -> Any:
        # If we're inside a running loop, avoid calling inner._run() which may try asyncio.run
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return _run_coro_in_new_loop(self._call_inner_arun(kwargs))

        # Prefer sync if available when no loop is running
        try:
            result = self.inner._run(**kwargs)
            # If inner incorrectly returns a coroutine, run it
            if inspect.isawaitable(result):
                return asyncio.run(result)
            return result
        except Exception:
            # Fallback to async path when sync invocation fails
            return asyncio.run(self._call_inner_arun(kwargs))

    async def _arun(self, **kwargs: Any) -> Any:
        # Delegate to inner async, adding config if required
        try:
            return await self._call_inner_arun(kwargs)
        except AttributeError:
            # Fallback to sync if async not available
            return self.inner._run(**kwargs)

    async def _call_inner_arun(self, kwargs: dict[str, Any]) -> Any:
        sig = None
        try:
            sig = inspect.signature(self.inner._arun)  # type: ignore[attr-defined]
        except Exception:
            pass
        if sig and "config" in sig.parameters and "config" not in kwargs:
            return await self.inner._arun(config={}, **kwargs)  # type: ignore[misc]
        return await self.inner._arun(**kwargs)  # type: ignore[misc]

    @property
    def tool_call_schema(self) -> dict:  # type: ignore[override]
        schema = getattr(self.inner, "tool_call_schema", None)
        if not isinstance(schema, dict):
            return {}
        # Filter properties to those present in args_schema
        props = schema.get("properties") or {}
        if not isinstance(props, dict):
            return {}
        allowed = set(getattr(self.args_schema, "model_fields", {}).keys())
        if not allowed:
            # Passthrough args: keep original schema as-is
            return dict(schema)
        filtered_props = {k: v for k, v in props.items() if k in allowed}
        new_schema = dict(schema)
        new_schema["properties"] = filtered_props
        # Optional: adjust required to only allowed
        required = schema.get("required") or []
        if isinstance(required, list):
            new_schema["required"] = [k for k in required if k in allowed]
        return new_schema


class _PassthroughArgs(BaseModel):
    # Accept any fields to match underlying tool signature without strict validation
    model_config = ConfigDict(extra="allow")


class MCPLoader:
    """Load LangChain tools from one or multiple MCP servers.

    connections: mapping of server name -> connection config accepted by MultiServerMCPClient.
    Example connection entries:
      {
        "math": {"command": "python", "args": ["/path/to/math_server.py"], "transport": "stdio"},
        "weather": {"url": "http://localhost:8000/mcp", "transport": "streamable_http"}
      }
    """

    def __init__(self, connections: dict[str, Any]) -> None:
        self._connections = connections

    async def aload(self) -> list[LoadedTool]:
        client = MultiServerMCPClient(self._connections)
        loaded: list[LoadedTool] = []
        try:
            # Fetch per-server to preserve origin metadata
            for server_name, cfg in self._connections.items():
                tools: list[BaseTool] = await client.get_tools(server_name=server_name)
                # Derive a stable, informative prefix for tool names based on server identity
                cfg_text = " ".join(
                    str(v) for v in [server_name, cfg.get("command"), cfg.get("url")] if v
                ).lower()
                if "ccxt" in cfg_text:
                    prefix = "ccxt"
                elif "mindmap" in cfg_text or "memory" in cfg_text:
                    prefix = "mindmap"
                else:
                    # Fallback to a sanitized server name
                    import re as _re

                    prefix = _re.sub(r"[^a-zA-Z0-9_-]", "_", server_name).strip("_") or "mcp"
                for t in tools:
                    desc = getattr(t, "description", "MCP tool") or "MCP tool"
                    # Always wrap to guarantee sync invocation, regardless of inner implementation
                    # Prefix the tool name for clearer attribution and easier selection heuristics
                    base_name = t.name or "tool"
                    prefixed_name = f"{prefix}_{base_name}" if not base_name.startswith(prefix) else base_name
                    wrapped = _SyncProxyTool(
                        name=prefixed_name,
                        description=desc,
                        inner=t,
                        args_schema=_PassthroughArgs,
                    )
                    loaded.append(
                        LoadedTool(
                            name=getattr(wrapped, "name", prefixed_name),
                            description=desc,
                            tool=wrapped,
                            source="mcp",
                            origin=server_name,
                        )
                    )
        finally:
            # Best-effort graceful shutdown
            close_coro = getattr(client, "aclose", None)
            if callable(close_coro):
                try:
                    await close_coro()  # type: ignore[misc]
                except Exception:
                    pass
        return loaded

    def load(self) -> list[LoadedTool]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Caller should use `await aload()` in async contexts
            raise RuntimeError(
                "Use 'await MCPLoader(...).aload()' within an async event loop"
            )
        return asyncio.run(self.aload())
