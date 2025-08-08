import os
import sys
from typing import Any

import pytest
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Ensure package is importable when running tests directly
ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from langtoolkit.builder import build_tool_hub


class SearchCategory:
    GENERAL = "general"


class SearchResult(dict):
    pass


class SearXNGClient:
    def __init__(self, host: str = "http://localhost:8080"):
        self.host = host.rstrip("/")

    def search(
        self, query: str, categories: list[str] | None = None
    ) -> list[SearchResult]:
        """
        Search for a query on the searxng web search engine
        Args:
            query: The query to search for
            categories: The categories to search in
        Returns:
            A list of search results
        """
        search_url = f"{self.host}/search"
        params = {"q": query, "format": "json"}
        if categories:
            params["categories"] = ",".join([f"{c}:" for c in categories])
        resp = requests.get(search_url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [SearchResult(r) for r in data.get("results", [])]


@pytest.mark.integration
def test_integration_agent_with_sdk_openapi_mcp():
    load_dotenv()
    mcp_url = "http://127.0.0.1:5555/mcp"

    sdk_client = SearXNGClient()
    sources: list[Any] = [
        sdk_client,
        {"local_mcp": {"url": mcp_url, "transport": "sse"}},
    ]
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    tasks = ["search for the best restaurant in the world", "what in the mind map?"]

    hub = build_tool_hub(sources, llm=llm)
    for task in tasks:
        tools = hub.query_tools(task, k=2)
        assert tools, "Unified hub returned no tools"
        tool_names = [t.name for t in tools]
        print("Selected tools:", tool_names)

        # Check expected tool selection per task intent
        if "mind map" in task.lower() or "mindmap" in task.lower():
            mindmap_keywords = ("mindmap", "mind_map", "concept", "mcp_memory")
            has_mindmap = any(
                any(kw in name.lower() for kw in mindmap_keywords)
                for name in tool_names
            )
            assert (
                has_mindmap
            ), "Expected a mind map tool (e.g., get_mindmap/add_concept) for mind map task"
        else:
            # Expect the SDK search tool from SearXNGClient to be present
            has_search = any("SearXNGClient__search" in name for name in tool_names)
            assert (
                has_search
            ), "Expected web search tool (SearXNGClient__search) for search task"

        agent = create_react_agent(llm, tools, verbose=True, strict=False)
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]}, config={"recursion_limit": 8}
        )
        assert isinstance(result, dict)
        print(result)
