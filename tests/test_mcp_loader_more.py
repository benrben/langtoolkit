import os
import sys
import asyncio

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from typing import Any
from pydantic import BaseModel, create_model

from langchain_core.tools import BaseTool

from langtoolkit.mcp_loader import MCPLoader, _SyncProxyTool


def test_sync_proxy_schema_filtering():
    class Inner(BaseTool):
        name: str = "t"
        description: str = "d"

        @property
        def tool_call_schema(self) -> dict:  # type: ignore[override]
            return {
                "type": "object",
                "properties": {"x": {"type": "int"}, "y": {"type": "int"}, "z": {"type": "int"}},
                "required": ["x", "z"],
            }

        def _run(self, **kwargs: Any) -> Any:
            return 1

    Args = create_model("Args", x=(int, ...), y=(int, ...))  # type: ignore
    wrapped = _SyncProxyTool(inner=Inner(), args_schema=Args)
    schema = wrapped.tool_call_schema
    assert set(schema.get("properties", {}).keys()) == {"x", "y"}
    assert schema.get("required") == ["x"]


def test_mcp_loader_prefix_fallback_and_load_methods(monkeypatch):
    class _AsyncTool(BaseTool):
        name: str = "run"
        description: str = "d"

        async def _arun(self, **kwargs: Any) -> Any:
            return 1

        def _run(self, **kwargs: Any) -> Any:
            return 1

    class _FakeClient:
        def __init__(self, mapping):
            self.mapping = mapping

        async def get_tools(self, server_name: str):
            return self.mapping.get(server_name, [])

        async def aclose(self):
            return None

    def _fake_client_ctor(conns):
        return _FakeClient({
            "Other": [_AsyncTool()],
        })

    monkeypatch.setattr("langtoolkit.mcp_loader.MultiServerMCPClient", _fake_client_ctor)

    loader = MCPLoader({"Other": {"command": "foo"}})
    loaded = asyncio.run(loader.aload())
    assert loaded and loaded[0].name.startswith("Other_") and loaded[0].origin == "Other"

    # load() outside loop should delegate to asyncio.run
    loaded2 = loader.load()
    assert isinstance(loaded2, list)

    # load() inside running loop must raise
    async def _runner():
        try:
            loader.load()
        except RuntimeError:
            return True
        return False

    assert asyncio.run(_runner()) is True


