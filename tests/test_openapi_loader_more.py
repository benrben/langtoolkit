import os
import sys
from unittest.mock import patch
import asyncio

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import requests

from langtoolkit.openapi_loader import EndpointTool, OpenAPILoader


def test_endpoint_tool_build_url_no_slashes():
    t = EndpointTool(
        name="x",
        description="",
        method="get",
        base_url="http://example.com",
        path_template="a/{id}",
    )
    url = t._build_url({"id": 2})
    assert url == "http://example.com/a/2"


def test_endpoint_tool_arun_and_request_exception():
    t = EndpointTool(
        name="x",
        description="",
        method="get",
        base_url="http://example.com",
        path_template="a/{id}",
    )

    class _ReqErr(requests.exceptions.RequestException):
        pass

    with patch("langtoolkit.openapi_loader.requests.request", side_effect=_ReqErr("boom")):
        out = asyncio.run(t._arun(path_params={"id": 1}))
        assert out["error"]


def test_openapi_loader_fetch_spec_and_skip_invalid_paths_and_methods():
    # Successful _fetch_spec path
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "openapi": "3.0.0",
                "servers": [{"url": "https://api.example.test"}],
                "paths": {
                    "/skip": "not a dict",
                    "/trace": {"trace": {}},
                    "/ok": {"get": {"summary": "OK"}},
                },
            }

    with patch("langtoolkit.openapi_loader.requests.get", return_value=_Resp()):
        tools = OpenAPILoader("https://api.example.test/openapi.json").load()
        # Only one valid tool should be produced
        names = [t.name for t in tools]
        assert any(name.startswith("get_") for name in names)


