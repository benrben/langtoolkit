from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

# Load environment variables from .env early
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from .mcp_loader import MCPLoader
from .openapi_loader import OpenAPILoader
from .sdk_loader import SDKLoader
from .tool import UnifiedToolHub


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


async def abuild_tool_hub(
    sources: Sequence[Any],
    *,
    llm: Any,
    embedding_model: Any | None = None,
) -> UnifiedToolHub:
    """Build a UnifiedToolHub from a list of sources.

    sources may contain:
      - str module names (e.g., "math", "os.path")
      - str OpenAPI spec URLs (e.g., "http://127.0.0.1:5555/openapi.json")
      - dict MCP connection map { server_name: connection }
    """
    hub = UnifiedToolHub(embedding_model=embedding_model)

    # Loaders that are synchronous
    for src in sources:
        if isinstance(src, str) and _is_url(src):
            # OpenAPI spec
            openapi_loader = OpenAPILoader(src, llm=llm)
            hub.add_loaded_tools(openapi_loader.load())
        elif isinstance(src, str):
            # SDK module name
            sdk_loader = SDKLoader(src)
            hub.add_loaded_tools(sdk_loader.load())
        elif isinstance(src, dict):
            # MCP handled in async block below
            continue
        else:
            # SDK object instance or module object
            sdk_loader = SDKLoader(src)
            hub.add_loaded_tools(sdk_loader.load())

    # Load MCP sources (async)
    for src in sources:
        if isinstance(src, dict):
            mcp_loader = MCPLoader(src)
            hub.add_loaded_tools(await mcp_loader.aload())

    return hub


def build_tool_hub(
    sources: Sequence[Any],
    *,
    llm: Any,
    embedding_model: Any | None = None,
) -> UnifiedToolHub:
    """Synchronous wrapper around abuild_tool_hub()."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "Use 'await abuild_tool_hub(...)' within an async event loop"
        )
    return asyncio.run(
        abuild_tool_hub(sources, llm=llm, embedding_model=embedding_model)
    )
