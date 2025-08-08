import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from typing import Any

from langchain_core.tools import BaseTool

from langtoolkit.mcp_loader import _PassthroughArgs, _SyncProxyTool


class _FakeAsyncTool(BaseTool):
    name: str = "fake_tool"
    description: str = "Fake async tool"

    # Simulate an inner tool schema that would be useful to preserve
    @property
    def tool_call_schema(self) -> dict:  # type: ignore[override]
        return {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x"],
        }

    async def _arun(self, **kwargs: Any) -> Any:
        await asyncio.sleep(0.01)
        return {"called_with": dict(kwargs)}

    # Provide sync fallback to satisfy abstract base
    def _run(self, **kwargs: Any) -> Any:
        return asyncio.run(self._arun(**kwargs))


def test_sync_proxy_preserves_schema_with_passthrough_args():
    inner = _FakeAsyncTool()
    wrapped = _SyncProxyTool(inner=inner, args_schema=_PassthroughArgs)
    # With passthrough (no allowed fields), the original schema should be kept
    schema = wrapped.tool_call_schema
    assert "properties" in schema and "x" in schema["properties"]


def test_sync_proxy_run_inside_running_loop():
    inner = _FakeAsyncTool()
    wrapped = _SyncProxyTool(inner=inner, args_schema=_PassthroughArgs)

    async def _runner():
        # Call sync _run from within a running loop; should not raise due to asyncio.run
        result = wrapped._run(x=1, y=2)
        assert result == {"called_with": {"x": 1, "y": 2}}

    asyncio.run(_runner())
