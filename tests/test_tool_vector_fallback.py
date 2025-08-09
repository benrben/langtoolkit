import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langchain_core.tools import BaseTool

from langtoolkit.tool import UnifiedToolHub, LoadedTool


def test_query_tools_falls_back_on_vector_failure(monkeypatch):
    class BadEmb:
        def embed_documents(self, texts):
            return [[0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            # Return mismatched dimension to trigger vector error
            return [1.0, 0.0, 0.0]

    class T(BaseTool):
        name: str = "sdk__search"
        description: str = "search"

        def _run(self, **kwargs):
            return 1

    hub = UnifiedToolHub(embedding_model=BadEmb())
    hub.add_loaded_tools([LoadedTool(name="sdk__search", description="d", tool=T(), source="sdk", origin="x")])

    # Should not raise due to vector fallback
    tools = hub.query_tools("anything", k=1)
    assert tools and getattr(tools[0], "name", "")


