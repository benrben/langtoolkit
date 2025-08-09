import os
import sys
import importlib
from typing import Any

import pytest
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Ensure package is importable when running tests directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ensure we import the local workspace version of `langtoolkit`, not any installed dist
for _m in list(sys.modules.keys()):
    if _m == "langtoolkit" or _m.startswith("langtoolkit."):
        del sys.modules[_m]

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

        agent = create_react_agent(llm, tools)
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]}, config={"recursion_limit": 8}
        )
        assert isinstance(result, dict)
        print(result)


@pytest.mark.integration
def test_integration_agent_with_sdk_and_two_mcps():
    """Integration test: SDK + two MCP servers (stdio and SSE).

    Skipped by default unless integration tests are enabled. Expects:
      - `mcp-server-ccxt` available on PATH, speaking MCP over stdio
      - A mind map MCP available at http://localhost:5555/mcp (SSE)
      - Optional SearXNG running at http://localhost:8080 for the SDK example
    """
    load_dotenv()

    _MCP_CONNECTIONS = {
        "crypto-search-tools": {
            "command": "mcp-server-ccxt",
            # Note: stdio transport; host/port are server args, not a web port for our app
            "args": ["--host", "127.0.0.1", "--port", "5000"],
            "transport": "stdio",
        },
        "MindMap-Memory": {
            "url": "http://localhost:5555/mcp",
            "transport": "sse",
        },
    }

    # SDK source (simple web search client)
    sdk_client = SearXNGClient()
    sources: list[Any] = [sdk_client, _MCP_CONNECTIONS]

    # Use any LangChain-compatible LLM; model name can be provided via env if desired
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"))

    tasks = [
        {"query": "search latest btc price and exchanges", "expected_tools": "ccxt"},
        {"query": "who is the best football player in the world?", "expected_tools": "SearXNGClient__search"},
        
    ]

    hub = build_tool_hub(sources, llm=llm)
    for task in tasks:
        tools = hub.query_tools(task, k=3)
        assert tools, "Unified hub returned no tools"
        tool_names = [t.name for t in tools]
        print("Selected tools:", tool_names)
        assert any(tool_name.startswith(task["expected_tools"]) for tool_name in tool_names), f"Expected tool {task['expected_tools']} not found"
        
        # Build a small agent over the selected tools and run the task
        agent = create_react_agent(llm, tools)
        # Ensure the human message content is a string for the LLM
        msg_content = task["query"] if isinstance(task, dict) else str(task)
        result = agent.invoke(
            {"messages": [HumanMessage(content=msg_content)]},
            config={"recursion_limit": 8},
        )
        for r in result:
            print(r)
            
