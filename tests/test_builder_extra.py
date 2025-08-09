import asyncio
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langchain_core.tools import BaseTool

# Force local workspace import of langtoolkit, not any installed version
for _m in list(sys.modules.keys()):
    if _m == "langtoolkit" or _m.startswith("langtoolkit."):
        del sys.modules[_m]

import importlib
import types

import langtoolkit.builder as builder
from langtoolkit.tool import LoadedTool


class _T(BaseTool):
    name: str = "t"
    description: str = ""

    def _run(self, **kwargs: Any) -> Any:  # pragma: no cover - trivial
        return 1


def test_abuild_tool_hub_routes_sources(monkeypatch):
    class _FakeOpenAPI:
        def __init__(self, spec_url: str, llm: Any) -> None:
            self.spec_url = spec_url
            self.llm = llm

        def load(self) -> list[LoadedTool]:
            return [
                LoadedTool(
                    name="openapi__get_x",
                    description="",
                    tool=_T(),
                    source="openapi",
                    origin=self.spec_url,
                )
            ]

    class _FakeSDK:
        def __init__(self, src: Any) -> None:
            self.src = src

        def load(self) -> list[LoadedTool]:
            return [
                LoadedTool(
                    name="sdk__search",
                    description="",
                    tool=_T(),
                    source="sdk",
                    origin=str(self.src),
                )
            ]

    class _FakeMCP:
        def __init__(self, connections: dict[str, Any]) -> None:
            self.connections = connections

        async def aload(self) -> list[LoadedTool]:
            return [
                LoadedTool(
                    name="mcp__tool",
                    description="",
                    tool=_T(),
                    source="mcp",
                    origin="fake",
                )
            ]

    # Override the loader classes directly on the imported module to avoid any
    # ambiguity about environments where an installed package may also exist.
    builder.OpenAPILoader = _FakeOpenAPI  # type: ignore[assignment]
    builder.SDKLoader = _FakeSDK  # type: ignore[assignment]
    builder.MCPLoader = _FakeMCP  # type: ignore[assignment]

    sources: list[Any] = [
        "http://example.com/openapi.json",
        "math",
        {"fake": {"command": "x"}},
    ]
    hub = asyncio.run(builder.abuild_tool_hub(sources, llm=None))
    names = sorted(t.name for t in hub.all_tools())
    assert names == ["mcp__tool", "openapi__get_x", "sdk__search"]


async def _call_build_tool_hub_inside_loop(monkeypatch):
    # Building inside a running loop should raise
    try:
        builder.build_tool_hub([], llm=None)
    except RuntimeError:
        return True
    return False


def test_build_tool_hub_raises_with_running_loop():
    # Use the real builder.build_tool_hub here
    assert asyncio.run(_call_build_tool_hub_inside_loop({})) is True


