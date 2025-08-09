import asyncio
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langchain_core.tools import BaseTool

from langtoolkit.mcp_loader import MCPLoader
from langtoolkit.tool import LoadedTool


class _AsyncTool(BaseTool):
    name: str = "n"
    description: str = "d"

    async def _arun(self, **kwargs: Any) -> Any:
        return 1


class _FakeClient:
    def __init__(self, mapping):
        self.mapping = mapping

    async def get_tools(self, server_name: str):  # pragma: no cover - called indirectly
        return self.mapping.get(server_name, [])

    async def aclose(self):
        return None


async def test_mcp_loader_prefixes_and_wraps(monkeypatch):
    def _fake_client_ctor(conns):
        return _FakeClient({
            "crypto-search-tools": [_AsyncTool(name="list-exchanges")],
            "MindMap-Memory": [_AsyncTool(name="get_mindmap")],
        })

    monkeypatch.setattr("langtoolkit.mcp_loader.MultiServerMCPClient", _fake_client_ctor)

    loader = MCPLoader({
        "crypto-search-tools": {"command": "mcp-server-ccxt"},
        "MindMap-Memory": {"url": "http://x/mcp"},
    })
    loaded = await loader.aload()
    names = sorted(t.name for t in loaded)
    assert names == ["ccxt_list-exchanges", "mindmap_get_mindmap"]



