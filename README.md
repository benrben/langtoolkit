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

## Realistic examples

The patterns below show how to use LangToolkit with real SDKs, a familiar OpenAPI spec, and multiple MCP servers. They also demonstrate natural-language tool selection via `query_tools(...)`.

### 0) SDK example: Python math

```python
from langtoolkit import build_tool_hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

hub = build_tool_hub(["math"], llm=None)

task = "compute sine of 1 radian and cosine of 0"
tools = hub.query_tools(task, k=3)
agent = create_react_agent(ChatOpenAI(), tools, verbose=True)
print(agent.invoke({"messages": [{"type": "human", "content": task}]}))
```

### 1) SDK example: S3 (boto3) passed directly

You can pass the S3 client object directly; its methods become tools.

```python
# pip install boto3
import boto3
from langtoolkit import build_tool_hub

s3 = boto3.client("s3")
hub = build_tool_hub([s3], llm=None)

# Discover a couple of tool names
print([t.name for t in hub.all_tools()][:8])

# LLM-driven execution (agent selects and runs tools)
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

task = "list my S3 buckets and upload 'hi' to my-bucket as hello.txt"
selected = hub.query_tools(task, k=4)
agent = create_react_agent(ChatOpenAI(), selected, verbose=True)
result = agent.invoke({"messages": [{"type": "human", "content": task}]})
print(result)
```

Run with an LLM (agent executes the selected tools):

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Select task-scoped tools based on intent
task = "upload a short note to s3 bucket my-bucket"
selected = hub.query_tools(task, k=2)

# Let the LLM plan and call the tools
llm = ChatOpenAI()  # requires OPENAI_API_KEY
agent = create_react_agent(llm, selected, verbose=True)
result = agent.invoke({"messages": [{"type": "human", "content": task}]})
print(result)
```

Notes:
- Configure AWS credentials via environment or local profile for boto3.
- Dynamic SDKs may have generic signatures; for stricter schemas, wrap selected methods in a small typed class.

### 2) OpenAPI example: Swagger Petstore

Use the public Petstore v3 example spec (`https://petstore3.swagger.io/api/v3/openapi.json`). Each OpenAPI operation becomes one tool.

```python
from langtoolkit import build_tool_hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

sources = ["https://petstore3.swagger.io/api/v3/openapi.json"]
hub = build_tool_hub(sources, llm=None)

task = "create a new pet named Fluffy, then fetch it by id"
tools = hub.query_tools(task, k=3)
agent = create_react_agent(ChatOpenAI(), tools, verbose=True)
result = agent.invoke({"messages": [{"type": "human", "content": task}]})
print(result)
```

### 3) Multiple MCP servers

Connect to more than one MCP server; LangToolkit will fetch and wrap their tools with event-loop–safe sync adapters.

```python
from langtoolkit import build_tool_hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

mcp_connections = {
    "notes": {"url": "http://127.0.0.1:5555/mcp", "transport": "sse"},
    # stdio example (Node): the process should speak MCP over stdio
    # "fs": {
    #   "command": "node",
    #   "args": ["/absolute/path/to/fs-server.mjs"],
    #   "transport": "stdio"
    # },
}
hub = build_tool_hub([mcp_connections], llm=None)

task = "in the mind map, add a node 'Vector Store' linked to 'Embeddings'"
tools = hub.query_tools(task, k=2)
agent = create_react_agent(ChatOpenAI(), tools, verbose=True)
print(agent.invoke({"messages": [{"type": "human", "content": task}]}))
```

#### MCP over stdio (explicit, runnable pattern)

If your MCP server is a local process that communicates over stdio, configure it with `command`/`args` and `transport: "stdio"`.

Node example:

```python
from langtoolkit import build_tool_hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

mcp_stdio = {
    "fs": {
        "command": "node",
        "args": ["/absolute/path/to/fs-server.mjs"],
        "transport": "stdio",
    }
}

hub = build_tool_hub([mcp_stdio], llm=None)
task = "list files under /tmp and read /tmp/hello.txt"
tools = hub.query_tools(task, k=3)
agent = create_react_agent(ChatOpenAI(), tools, verbose=True)
print(agent.invoke({"messages": [{"type": "human", "content": task}]}))
```

Python example:

```python
from langtoolkit import build_tool_hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

mcp_stdio = {
    "py_notes": {
        "command": "python",
        "args": ["/absolute/path/to/notes_mcp_server.py"],
        "transport": "stdio",
    }
}

hub = build_tool_hub([mcp_stdio], llm=None)
task = "create a note 'MCP stdio works' and then fetch it"
tools = hub.query_tools(task, k=3)
agent = create_react_agent(ChatOpenAI(), tools, verbose=True)
print(agent.invoke({"messages": [{"type": "human", "content": task}]}))
```

### 4) Mixed hub: S3 wrapper + Petstore + two MCPs

```python
from typing import Any
import boto3
from langtoolkit import build_tool_hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


class S3Tools:
    def __init__(self, client: Any | None = None) -> None:
        self.s3 = client or boto3.client("s3")

    def list_buckets(self) -> list[str]:
        resp = self.s3.list_buckets()
        return [b["Name"] for b in resp.get("Buckets", [])]

    def put_text(self, bucket: str, key: str, text: str) -> dict:
        self.s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
        return {"bucket": bucket, "key": key, "bytes": len(text)}


sources = [
    S3Tools(),
    "https://petstore3.swagger.io/api/v3/openapi.json",
    {
        "notes": {"url": "http://127.0.0.1:5555/mcp", "transport": "sse"},
        "knowledge": {"url": "http://127.0.0.1:6666/mcp", "transport": "sse"},
    },
]
hub = build_tool_hub(sources, llm=None)

llm = ChatOpenAI()
for task in [
    "list my s3 buckets",
    "upload 'build ok' to s3 bucket my-bucket as build.txt",
    "create a pet named Spike",
    "in the mind map, show concepts related to 'agents'",
]:
    tools = hub.query_tools(task, k=3)
    agent = create_react_agent(llm, tools, verbose=True)
    result = agent.invoke({"messages": [{"type": "human", "content": task}]})
    print(task, "->", result)
```

Tips:
- `query_tools` uses embeddings (OpenAI if `OPENAI_API_KEY` is set; otherwise local) plus keyword boosting.
- Tool names encode provenance (e.g., `sdk`, `openapi`, `mcp`) so you can attribute the choice or log it.

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