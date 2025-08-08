import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langtoolkit.sdk_loader import SDKLoader


def test_sdk_loader_wraps_math_and_generates_tools():
    loader = SDKLoader("math", max_tools_per_module=5)
    tools = loader.load()
    assert tools, "Expected some tools from math module"


def test_callable_tool_arun_offloads_sync_callable():
    class Obj:
        def heavy(self, x: int) -> int:
            total = 0
            for i in range(20000):
                total += (i * x) % 7
            return total

    loader = SDKLoader(Obj(), include_predicate=lambda n, f: n == "heavy")
    loaded = loader.load()
    tool = loaded[0].tool

    async def _runner():
        # Should not block the event loop; just sanity run
        res = await tool._arun(x=3)
        assert isinstance(res, int)

    asyncio.run(_runner())
