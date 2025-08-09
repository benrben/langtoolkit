import os
import sys
import asyncio

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from typing import Any

from langchain_core.tools import BaseTool

from langtoolkit.mcp_loader import MCPLoader, _SyncProxyTool, _PassthroughArgs


def test_sync_proxy_model_post_init_defaults_and_exceptions():
    class Weird(BaseTool):
        name: str = ""
        description: str = ""

        def _run(self, **kwargs: Any) -> Any:
            return 1

    # Defaults should be applied from inner (empty strings are allowed)
    wrapped = _SyncProxyTool(inner=Weird(), args_schema=_PassthroughArgs)
    assert wrapped.name == "" and wrapped.description == ""


def test_sync_proxy_run_awaitable_and_fallback():
    class ReturnCoro(BaseTool):
        name: str = "t"
        description: str = "d"

        def _run(self, **kwargs: Any) -> Any:
            async def _coro():
                return {"ok": True}

            return _coro()

        async def _arun(self, **kwargs: Any) -> Any:
            return {"ok": True}

    w1 = _SyncProxyTool(inner=ReturnCoro(), args_schema=_PassthroughArgs)
    assert w1._run() == {"ok": True}

    class RunRaises(BaseTool):
        name: str = "t2"
        description: str = "d2"

        def _run(self, **kwargs: Any) -> Any:
            raise RuntimeError("x")

        async def _arun(self, **kwargs: Any) -> Any:
            return 2

    w2 = _SyncProxyTool(inner=RunRaises(), args_schema=_PassthroughArgs)
    assert w2._run() == 2


def test_sync_proxy_arun_fallback_to_sync_when_no_arun():
    class SyncOnly(BaseTool):
        name: str = "t"
        description: str = "d"

        def _run(self, **kwargs: Any) -> Any:
            return 42

    w = _SyncProxyTool(inner=SyncOnly(), args_schema=_PassthroughArgs)

    async def _runner():
        res = await w._arun()
        assert res == 42

    asyncio.run(_runner())


def test_sync_proxy_arun_injects_config():
    class NeedsConfig(BaseTool):
        name: str = "t"
        description: str = "d"

        async def _arun(self, *, config: dict, x: int) -> Any:  # type: ignore[override]
            return ("ok", config, x)

    w = _SyncProxyTool(inner=NeedsConfig(), args_schema=_PassthroughArgs)

    async def _runner():
        res = await w._arun(x=3)
        assert res[0] == "ok" and isinstance(res[1], dict) and res[2] == 3

    asyncio.run(_runner())


def test_mcp_loader_prefixes_edgecases_and_close_error(monkeypatch):
    class Tool(BaseTool):
        def __init__(self, name: str, description: str = ""):
            super().__init__(name=name, description=description)  # type: ignore[call-arg]

        def _run(self, **kwargs: Any) -> Any:
            return 1

    class _FakeClient:
        def __init__(self, mapping):
            self.mapping = mapping

        async def get_tools(self, server_name: str):
            return self.mapping.get(server_name, [])

        async def aclose(self):
            raise RuntimeError("close error")

    def _fake_client_ctor(conns):
        return _FakeClient(
            {
                "crypto-search-tools": [Tool(name="ccxt_balance", description="")],
                "MindMap-Memory": [Tool(name="get_mindmap", description="mm")],
                "Other": [Tool(name="run", description="r")],
            }
        )

    monkeypatch.setattr("langtoolkit.mcp_loader.MultiServerMCPClient", _fake_client_ctor)

    loader = MCPLoader(
        {
            "crypto-search-tools": {"command": "mcp-server-ccxt"},
            "MindMap-Memory": {"url": "http://x/mcp"},
            "Other": {"command": "foo"},
        }
    )
    loaded = asyncio.run(loader.aload())
    names = {t.name for t in loaded}
    assert "ccxt_balance" in names  # already prefixed; no double prefix
    assert "mindmap_get_mindmap" in names
    assert any(n.startswith("Other_") for n in names)
    # description default used when empty
    descs = {t.description for t in loaded}
    assert "MCP tool" in descs


