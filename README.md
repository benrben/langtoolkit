# LangToolkit

Unified tool interface for SDKs, OpenAPI, and MCP – with a simple API and a focus on local, fast iteration.

## Requirements

- Python 3.11 or newer (tested with 3.13)
- macOS/Linux or WSL2

## Setup (venv)

Create an isolated environment and install dependencies from this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests (unit tests only by default):

```bash
pytest -q
```

Optional checks:

```bash
ruff check .
mypy langtoolkit
```

## What it does

LangToolkit loads “tools” from three kinds of sources and exposes them through a single hub:

- SDK: Introspects Python modules/objects into tools (e.g., `math.sin` → a callable tool)
- OpenAPI: One tool per API operation
- MCP: Discovers tools from MCP servers and adapts async tools for sync usage

## Quickstart

Build a hub from the Python `math` module and search the available tools:

```python
from langtoolkit import build_tool_hub

# llm is optional for SDK-only usage
hub = build_tool_hub(["math"], llm=None)

print("Some tool names:", [t.name for t in hub.all_tools()][:8])

# Retrieve the top tool candidates for a query
tools = hub.query_tools("sine of angle", k=2)
for t in tools:
    print(t.name, "-", t.description)
```

Combine SDK, OpenAPI, and MCP in one hub:

```python
from langtoolkit import build_tool_hub

sources = [
    "math",  # SDK
    "http://127.0.0.1:5555/openapi.json",  # OpenAPI
    {"local_mcp": {"url": "http://127.0.0.1:5555/mcp", "transport": "sse"}},  # MCP
]

hub = build_tool_hub(sources, llm=None)
print("Loaded", len(hub.all_tools()), "tools")
```

### Custom SDK + MCP example (like `tests/test_integration_agent.py`)

This shows how a custom Python client becomes tools automatically, and how to query tool selection and layout.

Requirements:
- Optional SearXNG running locally for web search: `http://localhost:8080`
- Optional MCP server: `http://127.0.0.1:5555/mcp` (SSE transport)

```python
import requests
from typing import Any

from langtoolkit import build_tool_hub


class SearchCategory:
    GENERAL = "general"


class SearchResult(dict):
    pass


class SearXNGClient:
    def __init__(self, host: str = "http://localhost:8080"):
        self.host = host.rstrip("/")

    def search(self, query: str, categories: list[str] | None = None):
        """
        Search for a query on the SearXNG web search engine
        Returns: A list of search results
        """
        search_url = f"{self.host}/search"
        params: dict[str, Any] = {"q": query, "format": "json"}
        if categories:
            params["categories"] = ",".join([f"{c}:" for c in categories])
        resp = requests.get(search_url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [SearchResult(r) for r in data.get("results", [])]


# Build sources with a custom SDK object and an optional MCP server
sdk_client = SearXNGClient()
sources: list[Any] = [
    sdk_client,
    {"local_mcp": {"url": "http://127.0.0.1:5555/mcp", "transport": "sse"}},
]

hub = build_tool_hub(sources, llm=None)

# Inspect the tool layout (names only)
tool_names = sorted(t.name for t in hub.all_tools())
print("Tools (sample):", tool_names[:10])

# Query for the best-matching tools per task intent
tasks = [
    "search for the best restaurant in the world",
    "what in the mind map?",
]
for task in tasks:
    tools = hub.query_tools(task, k=2)
    names = [t.name for t in tools]
    print(f"Task: {task}\nSelected tools: {names}\n")

    # Simple assertions (like the test):
    if "mind map" in task.lower() or "mindmap" in task.lower():
        mindmap_keywords = ("mindmap", "mind_map", "concept", "mcp_memory")
        has_mindmap = any(any(kw in n.lower() for kw in mindmap_keywords) for n in names)
        assert has_mindmap, "Expected a mind map tool for the mind map task"
    else:
        # Expect the SDK search tool from SearXNGClient to be present
        assert any("SearXNGClient__search" in n for n in names), "Expected web search tool"
```

If you want an agent, you can plug the hub into LangGraph (requires an LLM):

```python
from langgraph.prebuilt import create_react_agent
# e.g., from langchain_openai import ChatOpenAI; llm = ChatOpenAI()
llm = ...  # provide any LangChain-compatible LLM
agent = create_react_agent(llm, hub.all_tools(), verbose=True)
result = agent.invoke({"messages": [{"type": "human", "content": "find sine of 1"}]})
```

## API Surface

- `SDKLoader(modules)` → List of tools from modules/objects
- `OpenAPILoader(spec_url)` → List of tools derived from an OpenAPI spec
- `MCPLoader(connections)` → List of tools fetched from MCP servers
- `build_tool_hub(sources, llm=None, embedding_model=None)` → Unified hub
- `UnifiedToolHub.query_tools(query, k=5)` → Rank tools for a natural-language query
- `UnifiedToolHub.all_tools()` → All tools in the hub

Notes:

- The hub selects an embedding backend automatically:
  - OpenAI embeddings if `OPENAI_API_KEY` is set
  - A local sentence-transformers backend if available
  - A deterministic fallback embedding otherwise
- MCP tools are wrapped to work in sync contexts even if they are async internally.

## Running integration tests

Integration tests are skipped by default. Enable them with:

```bash
pytest -m integration -q
```

They expect:

- An MCP server reachable at `http://127.0.0.1:5555/mcp` (SSE transport)
- An LLM if you exercise agent examples (e.g., set `OPENAI_API_KEY` for OpenAI)

## License

MIT — see `LICENSE`.