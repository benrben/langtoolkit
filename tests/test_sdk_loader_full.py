import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import types

from langtoolkit.sdk_loader import SDKLoader


def test_sdk_loader_exclude_private_and_predicate_and_truncation():
    class Obj:
        def _hidden(self):
            return 0

        def a(self, x: int) -> int:
            return x

        def b(self) -> int:
            return 1

    loader = SDKLoader(
        Obj(),
        include_predicate=lambda name, func: name != "b",
        exclude_private=True,
        max_tools_per_module=1,
    )
    tools = loader.load()
    names = [t.name for t in tools]
    # "b" filtered and private skipped; with truncation only one tool remains
    assert len(names) == 1 and any("__a" in n for n in names)


