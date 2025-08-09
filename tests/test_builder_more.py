import os
import sys
import asyncio

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import types

import langtoolkit.builder as builder


class _Obj:
    def go(self, x: int) -> int:
        return x + 1


def test_build_tool_hub_runs_outside_loop():
    hub = builder.build_tool_hub([], llm=None)
    assert hub is not None


def test_abuild_tool_hub_accepts_object_instance():
    obj = _Obj()
    hub = asyncio.run(builder.abuild_tool_hub([obj], llm=None))
    # Should yield at least one tool from the object instance
    names = sorted(getattr(t, "name", "") for t in hub.all_tools())
    assert names and all(isinstance(n, str) for n in names)


