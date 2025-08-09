import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langtoolkit.sdk_loader import SDKLoader


def test_sdk_loader_instance_naming_uses_class():
    class Klass:
        def search(self, q: str) -> str:
            return q

    loaded = SDKLoader(Klass(), include_predicate=lambda n, f: n == "search").load()
    names = [t.name for t in loaded]
    assert any(name.startswith("Klass__search") for name in names)



