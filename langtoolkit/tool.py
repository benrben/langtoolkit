from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore

try:
    # Prefer OpenAI embeddings if configured; users can inject their own
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAIEmbeddings = None  # type: ignore

try:
    # Strong local default when OpenAI is not configured
    from langchain_community.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddings,  # type: ignore
    )
except Exception:  # pragma: no cover - optional dependency at runtime
    SentenceTransformerEmbeddings = None  # type: ignore


class _HashEmbedding:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for i, ch in enumerate(text):
            vec[(i + ord(ch)) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class LoadedTool:
    """A tool plus minimal metadata needed for indexing and attribution."""

    name: str
    description: str
    tool: BaseTool
    source: str
    origin: str


class UnifiedToolHub:
    """Aggregate tools from SDK, OpenAPI, and MCP; index and retrieve by query."""

    def __init__(
        self,
        *,
        embedding_model: Any | None = None,
    ) -> None:
        if embedding_model is None:
            # Choose the strongest available embedding backend in priority order
            openai_key = os.getenv("OPENAI_API_KEY")
            if OpenAIEmbeddings is not None and openai_key:
                embedding_model = OpenAIEmbeddings()
            elif SentenceTransformerEmbeddings is not None:
                # Default lightweight sentence-transformer model
                embedding_model = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
            else:
                # Deterministic fallback that requires no external deps or network
                embedding_model = _HashEmbedding()

        self._embedding = embedding_model
        self._vectorstore = InMemoryVectorStore.from_documents(
            documents=[], embedding=self._embedding
        )
        self._name_to_tool: dict[str, BaseTool] = {}
        # Cache the indexed text for each tool to enable keyword boosting at query time
        self._tool_text: dict[str, str] = {}

    def add_loaded_tools(self, items: Iterable[LoadedTool]) -> None:
        new_items = list(items)
        # Index immediately for simplicity
        documents: list[Document] = []
        # Ensure uniqueness of tool names by auto-suffixing on collision
        for lt in new_items:
            tool_name = lt.name
            if tool_name in self._name_to_tool:
                base = tool_name
                idx = 2
                while tool_name in self._name_to_tool:
                    tool_name = f"{base}_{idx}"
                    idx += 1
            self._name_to_tool[tool_name] = lt.tool
            text = f"name: {tool_name}\nsource: {lt.source}\norigin: {lt.origin}\n{lt.description}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "tool_name": tool_name,
                        "source": lt.source,
                        "origin": lt.origin,
                    },
                )
            )
            self._tool_text[tool_name] = text
        if documents:
            self._vectorstore.add_documents(documents)

    def all_tools(self) -> list[BaseTool]:
        return list(self._name_to_tool.values())

    def query_tools(self, query: str, k: int = 5) -> list[BaseTool]:
        if not self._name_to_tool:
            return []
        # Retrieve a wider candidate pool via vector similarity
        candidate_pool_size = min(max(20, k * 3), len(self._name_to_tool))
        vec_results = self._vectorstore.similarity_search(query, k=candidate_pool_size)
        vec_rank: dict[str, int] = {}
        for idx, doc in enumerate(vec_results):
            tname = doc.metadata.get("tool_name")
            if tname:
                vec_rank.setdefault(tname, idx)

        # Lightweight keyword boosting by matching query tokens against tool name/description
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t]

        def boost_for(tool_name: str) -> int:
            name_l = tool_name.lower()
            text_l = self._tool_text.get(tool_name, "").lower()
            name_hits = sum(1 for t in tokens if t in name_l)
            text_hits = sum(1 for t in tokens if t in text_l)
            return 2 * name_hits + text_hits

        # Combine scores: prioritize keyword matches, then vector rank
        all_names = list(self._name_to_tool.keys())
        scored = []
        default_vec_rank = len(self._name_to_tool) + 1000
        for name in all_names:
            b = boost_for(name)
            vr = vec_rank.get(name, default_vec_rank)
            # Higher boost is better; lower vec rank index is better. Use negative vr to sort ascending.
            scored.append((b, -vr, name))

        scored.sort(reverse=True)
        top_names = [name for (_b, _vr, name) in scored[:k]]
        return [self._name_to_tool[n] for n in top_names]
