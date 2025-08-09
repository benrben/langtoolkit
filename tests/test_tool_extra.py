import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langtoolkit.tool import UnifiedToolHub


def test_normalize_query_variants():
    hub = UnifiedToolHub(embedding_model=None)
    # Private method usage in tests to cover branches
    n = hub._normalize_query  # type: ignore[attr-defined]
    assert n("hello") == "hello"
    assert n({"query": "q"}) == "q"
    assert n({"text": "q2"}) == "q2"
    assert n({"prompt": "q3"}) == "q3"
    assert n({"content": "q4"}) == "q4"
    assert n({"messages": [{"content": "q5"}]}) == "q5"
    # Fallback string conversion
    assert n(123) == "123"



