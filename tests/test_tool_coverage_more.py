import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import types

from langchain_core.tools import BaseTool

import langtoolkit.tool as tool_mod
from langtoolkit.tool import UnifiedToolHub, LoadedTool, _HashEmbedding  # type: ignore


def test_embedding_selection_openai_and_sentence_and_fallback(monkeypatch):
    # 1) OpenAI path when key present and class available
    class _DummyEmb:
        def embed_documents(self, texts):
            return [[0.0] * 3 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 3

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(tool_mod, "OpenAIEmbeddings", _DummyEmb, raising=True)
    hub = UnifiedToolHub()
    assert isinstance(hub._embedding, _DummyEmb)

    # 2) SentenceTransformer path when OpenAI disabled
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _DummyST:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed_documents(self, texts):
            return [[0.0] * 3 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 3

    monkeypatch.setattr(tool_mod, "OpenAIEmbeddings", None, raising=False)
    monkeypatch.setattr(tool_mod, "SentenceTransformerEmbeddings", _DummyST, raising=True)
    hub2 = UnifiedToolHub()
    assert isinstance(hub2._embedding, _DummyST) and getattr(hub2._embedding, "model_name", "").endswith("MiniLM-L6-v2")

    # 3) Fallback hash embedding when neither is available
    monkeypatch.setattr(tool_mod, "OpenAIEmbeddings", None, raising=False)
    monkeypatch.setattr(tool_mod, "SentenceTransformerEmbeddings", None, raising=False)
    hub3 = UnifiedToolHub()
    assert isinstance(hub3._embedding, _HashEmbedding)

    # Cover _HashEmbedding methods directly
    he = _HashEmbedding(dim=16)
    vecs = he.embed_documents(["a", "b"])
    q = he.embed_query("c")
    assert len(vecs) == 2 and len(q) == 16


def test_normalize_query_corner_cases_and_query_tools_boosting(monkeypatch):
    class DummyEmb:
        def embed_documents(self, texts):
            return [[0.0] * 3 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 3

    hub = UnifiedToolHub(embedding_model=DummyEmb())

    class T(BaseTool):
        name: str = "sdk__search"
        description: str = "search tool"

        def _run(self, **kwargs):
            return 1

    class T2(BaseTool):
        name: str = "ccxt_get_price"
        description: str = "crypto exchanges"

        def _run(self, **kwargs):
            return 1

    hub.add_loaded_tools(
        [
            LoadedTool(name="sdk__search", description="search", tool=T(), source="sdk", origin="a"),
            LoadedTool(name="ccxt_get_price", description="crypto", tool=T2(), source="mcp", origin="b"),
        ]
    )

    # messages present but not dict -> fallback to string conversion
    q = hub._normalize_query({"messages": [1, 2, 3]})
    assert isinstance(q, str)

    # Boosting for general queries should surface search tools
    # Use a non-zero query embedding to avoid all-zero cosine NaNs
    class NonZeroEmb(DummyEmb):
        def embed_query(self, text):
            return [1.0, 0.0, 0.0]
        def embed_documents(self, texts):
            # Ensure vectors are unit-like to avoid zero norms
            return [[1.0, 0.0, 0.0] for _ in texts]

    hub = UnifiedToolHub(embedding_model=NonZeroEmb())
    hub.add_loaded_tools(
        [
            LoadedTool(name="sdk__search", description="search", tool=T(), source="sdk", origin="a"),
            LoadedTool(name="ccxt_get_price", description="crypto", tool=T2(), source="mcp", origin="b"),
        ]
    )

    top = hub.query_tools("who is the best?", k=1)
    assert top and getattr(top[0], "name", "").startswith("sdk__search")

    # Crypto terms should prefer ccxt tools
    top2 = hub.query_tools("latest btc exchange rates", k=1)
    assert top2 and getattr(top2[0], "name", "").startswith("ccxt_")


