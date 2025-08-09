from __future__ import annotations

import re
from typing import Any

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from .tool import LoadedTool


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


class EndpointArgs(BaseModel):
    # Allow population by both field name and alias
    model_config = ConfigDict(populate_by_name=True)

    path_params: dict[str, Any] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)
    # Avoid shadowing BaseModel.json by using a different name with alias "json"
    json_body: dict[str, Any] | None = Field(default=None, alias="json")
    headers: dict[str, Any] = Field(default_factory=dict)


class EndpointTool(BaseTool):
    name: str
    description: str
    method: str
    base_url: str
    path_template: str
    timeout: float = 30.0
    args_schema: type = EndpointArgs

    def _build_url(self, path_params: dict[str, Any]) -> str:
        url_path = self.path_template
        for key, val in path_params.items():
            url_path = url_path.replace("{" + str(key) + "}", str(val))
        # Ensure single slash join
        if self.base_url.endswith("/") and url_path.startswith("/"):
            return self.base_url[:-1] + url_path
        if not self.base_url.endswith("/") and not url_path.startswith("/"):
            return self.base_url + "/" + url_path
        return self.base_url + url_path

    def _run(
        self,
        path_params: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ) -> Any:
        path_params = path_params or {}
        query = query or {}
        headers = headers or {}
        url = self._build_url(path_params)
        try:
            resp = requests.request(
                self.method.upper(),
                url,
                params=query,
                json=json_body,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "method": self.method.upper(), "url": url}
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception:
            # Return raw text on non-JSON or error for transparency
            return {"status": resp.status_code, "text": resp.text, "url": url}

    async def _arun(self, **kwargs: Any) -> Any:
        # Simple sync adapter for now
        return self._run(**kwargs)


class OpenAPILoader:
    """Load one tool per OpenAPI operation with real names (operationId or method_path).

    These tools directly call the endpoint via requests with params:
      - path_params: dict for templated path parameters
      - query: dict for query string
      - json: dict for JSON body (POST/PUT/PATCH)
      - headers: optional headers
    """

    def __init__(
        self,
        spec_url: str,
        *,
        llm: Any = None,
        requests_timeout: float | None = 30.0,
    ) -> None:
        self._spec_url = spec_url
        self._timeout = requests_timeout or 30.0

    def _fetch_spec(self) -> dict:
        resp = requests.get(self._spec_url, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def load(self) -> list[LoadedTool]:
        spec = self._fetch_spec()
        servers = spec.get("servers") or []
        base_url = servers[0].get("url") if servers else ""
        if not base_url:
            # Fallback: attempt to derive origin from spec URL
            try:
                import urllib.parse as _urlparse

                parsed = _urlparse.urlparse(self._spec_url)
                if parsed.scheme and parsed.netloc:
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
            except Exception:  # pragma: no cover - defensive guard for malformed URLs
                base_url = ""
        paths = spec.get("paths", {})
        loaded: list[LoadedTool] = []
        used_names: set[str] = set()

        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, op in methods.items():
                if method.lower() not in {
                    "get",
                    "post",
                    "put",
                    "delete",
                    "patch",
                    "head",
                    "options",
                }:
                    continue
                op = op or {}
                op_id = op.get("operationId")
                raw_name = op_id or f"{method.lower()}_{path}"
                base_name = _sanitize_name(raw_name)
                name = base_name
                suffix = 2
                while name in used_names:
                    name = f"{base_name}_{suffix}"
                    suffix += 1
                used_names.add(name)
                desc = (
                    op.get("summary")
                    or op.get("description")
                    or f"{method.upper()} {path}"
                )

                tool = EndpointTool(
                    name=name,
                    description=desc,
                    method=method.lower(),
                    base_url=base_url or "",
                    path_template=path,
                    timeout=self._timeout,
                )
                loaded.append(
                    LoadedTool(
                        name=name,
                        description=desc,
                        tool=tool,
                        source="openapi",
                        origin=self._spec_url,
                    )
                )

        return loaded
