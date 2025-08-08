from .builder import abuild_tool_hub, build_tool_hub
from .sdk_loader import SDKLoader
from .tool import LoadedTool, UnifiedToolHub


# Optional imports: expose placeholders that raise informative ImportError only when used
def _optional_openapi_loader():
    try:
        from .openapi_loader import OpenAPILoader  # type: ignore

        return OpenAPILoader
    except Exception:  # pragma: no cover - executed only when extra is missing

        class _MissingOpenAPILoader:  # type: ignore
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "OpenAPILoader requires optional dependency 'requests'. "
                    "Install with: pip install 'langtoolkit[openapi]'"
                )

        return _MissingOpenAPILoader


def _optional_mcp_loader():
    try:
        from .mcp_loader import MCPLoader  # type: ignore

        return MCPLoader
    except Exception:  # pragma: no cover - executed only when extra is missing

        class _MissingMCPLoader:  # type: ignore
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "MCPLoader requires optional dependencies 'langchain-mcp-adapters' and 'aiohttp'. "
                    "Install with: pip install 'langtoolkit[mcp]'"
                )

        return _MissingMCPLoader


OpenAPILoader = _optional_openapi_loader()
MCPLoader = _optional_mcp_loader()

__version__ = "0.1.0"

__all__ = [
    "UnifiedToolHub",
    "LoadedTool",
    "build_tool_hub",
    "abuild_tool_hub",
    "SDKLoader",
    "OpenAPILoader",
    "MCPLoader",
    "__version__",
]
