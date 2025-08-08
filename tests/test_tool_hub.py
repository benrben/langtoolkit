import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langchain_core.tools import BaseTool

from langtoolkit.tool import LoadedTool, UnifiedToolHub


class _T(BaseTool):
    name: str = "t"
    description: str = ""

    def _run(self, **kwargs):
        return 1


def test_hub_deduplicates_tool_names():
    # Provide minimal dummy embedding model interface so we do not need OpenAI
    class DummyEmbeddings:
        def embed_documents(self, texts):
            return [[0.0] * 3 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 3

    hub = UnifiedToolHub(embedding_model=DummyEmbeddings())
    t = _T()
    items = [
        LoadedTool(name="dup", description="", tool=t, source="sdk", origin="a"),
        LoadedTool(name="dup", description="", tool=t, source="sdk", origin="b"),
    ]
    hub.add_loaded_tools(items)
    # Ensure map contains unique tool names after collision resolution
    assert len(set(hub._name_to_tool.keys())) == len(hub._name_to_tool.keys())
