import os
import sys
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import requests

from langtoolkit.openapi_loader import EndpointTool, OpenAPILoader


def test_endpoint_tool_build_url_and_text_fallback():
    t = EndpointTool(
        name="x",
        description="",
        method="get",
        base_url="http://example.com/",
        path_template="/a/{id}",
    )
    url = t._build_url({"id": 1})
    assert url == "http://example.com/a/1"
    # Non-JSON response returns text/status
    class _Resp:
        status_code = 200
        text = "ok"

        def raise_for_status(self):
            return None

        def json(self):  # pragma: no cover - exercised via except branch
            raise ValueError("not json")

    with patch("langtoolkit.openapi_loader.requests.request", return_value=_Resp()):
        out = t._run()
        assert out["text"] == "ok"


def test_openapi_loader_baseurl_from_spec_url():
    spec = {"openapi": "3.0.0", "paths": {"/a": {"get": {"summary": "A"}}}}
    with patch.object(OpenAPILoader, "_fetch_spec", return_value=spec):
        tools = OpenAPILoader("http://api.example.com/openapi.json").load()
        assert tools and tools[0].tool.base_url == "http://api.example.com"



