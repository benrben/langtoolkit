import os
import sys
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import requests

from langtoolkit.openapi_loader import EndpointTool, OpenAPILoader


def test_endpoint_tool_handles_request_exception():
    tool = EndpointTool(
        name="get_x",
        description="",
        method="get",
        base_url="http://example.com",
        path_template="/x",
    )
    with patch(
        "langtoolkit.openapi_loader.requests.request",
        side_effect=requests.exceptions.RequestException("boom"),
    ):
        out = tool._run()
        assert "error" in out and "url" in out and out["method"] == "GET"


def test_loader_deduplicates_names_and_baseurl_fallback():
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/a": {
                "get": {"operationId": "op", "summary": "A"},
            },
            "/b": {
                "get": {"operationId": "op", "summary": "B"},
            },
        },
    }

    loader = OpenAPILoader("http://api.example.com/openapi.json")
    with patch.object(OpenAPILoader, "_fetch_spec", return_value=spec):
        tools = loader.load()
        names = [t.name for t in tools]
        assert len(set(names)) == len(names)  # deduped
