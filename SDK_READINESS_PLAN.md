### LangToolkit Production-Ready SDK Plan (No CLI)

- **Objective**: Ship LangToolkit as a stable, pip-installable Python SDK with a minimal core, optional extras, strong tests, and CI/CD. No CLI.

### Goals
- **Production-ready SDK**: clean public API, semantic versioning, and documented behavior.
- **Minimal core deps**: optional extras for OpenAPI, MCP, embeddings, and agent examples.
- **Reliable**: comprehensive unit tests; integration tests opt-in.
- **Automated**: CI for lint/type/tests; release pipeline to PyPI.

### Public API stabilization
- **Public surface (re-exported in `langtoolkit.__init__`)**:
  - `UnifiedToolHub`, `LoadedTool`, `SDKLoader`, `OpenAPILoader`, `MCPLoader`, `build_tool_hub`, `abuild_tool_hub`.
  - Add `__version__` and keep SemVer discipline.
- **Deprecations**:
  - Optional: keep `build_toolkit = build_tool_hub` as a backwards-compat alias, document deprecation timeline; or remove and update docs to `build_tool_hub` only.
- **Quality**:
  - Add docstrings and precise type hints for all public classes/functions.
  - Introduce focused exceptions: `ToolLoadError`, `OpenAPISpecError`, `MCPClientError`.

### Packaging and layout
- **Adopt src layout**:
  - Move code into `src/langtoolkit/`; tests remain in `tests/`.
- **`pyproject.toml`**:
  - Metadata: `readme`, `license`, `authors`, `urls`, `classifiers`, `keywords`.
  - Versioning: consider `hatch-vcs` (recommended) or keep manual bumps.
  - Core dependencies (minimal):
    - `langchain-core>=0.2,<0.4`
    - `pydantic>=2,<3`
  - Optional extras:
    - `openapi`: `requests>=2.31`
    - `mcp`: `langchain-mcp-adapters>=0.1`, `aiohttp>=3.8`
    - `openai`: `langchain-openai>=0.1`
    - `embeddings-sentencetransformers`: `sentence-transformers>=2.2`, `numpy>=1.24`
    - `agent-examples`: `langgraph>=0.1`, `langchain-openai>=0.1`
    - `dev`: `pytest`, `pytest-asyncio`, `pytest-cov`, `mypy`, `ruff`, `black`, `types-requests`
  - Remove test/example deps from core install. Keep wheels small; do not ship CLI entry points.

### Runtime behavior hardening
- **Logging**: use `logging`; no prints. Respect `LOGLEVEL`/root config.
- **Timeouts**: keep `EndpointTool` timeouts; document defaults and how to override.
- **Event loop safety**: keep `_SyncProxyTool` and `_CallableTool._arun` safeguards; ensure tests prevent regressions.
- **Resource cleanup**: ensure `MCPLoader` closes clients (`aclose()` best-effort) and is covered by tests.

### Documentation
- **Installation**:
  - Core: `pip install langtoolkit`
  - OpenAPI tools: `pip install "langtoolkit[openapi]"`
  - MCP tools: `pip install "langtoolkit[mcp]"`
  - Embeddings (local): `pip install "langtoolkit[embeddings-sentencetransformers]"`
  - Agent examples: `pip install "langtoolkit[agent-examples]"`
- **Usage**:
  - Update examples to use `build_tool_hub` and current module paths.
  - Provide minimal snippets for each loader and for `UnifiedToolHub.query_tools`.
- **Env**:
  - Document optional `OPENAI_API_KEY`; clarify no network calls at import time.
- Add `CHANGELOG.md` and `CONTRIBUTING.md`. Document deprecation policy and SemVer.

### Tests
- **Default (unit) — run on CI**:
  - `SDKLoader`:
    - No/partial type hints; ignores `*args/**kwargs`; object methods; `include_predicate`; `exclude_private`; name dedupe; `_arun` thread offload.
  - `OpenAPILoader`:
    - Spec parsing; opId dedupe; base URL fallback; path param expansion; query/header forwarding; JSON vs non-JSON; request exception path.
  - `MCPLoader`:
    - Multi-server discovery; passthrough schema preservation; sync-call from running loop; client shutdown; error propagation.
  - `UnifiedToolHub`:
    - Collision handling; embedding fallbacks; keyword boosting affects order; empty hub; custom embedding injection; deterministic results with hash embedding.
  - `builder`:
    - Mixed sources; sync vs async guard (raises in running loop for sync wrapper); custom embedding model passthrough.
- **Integration (opt-in)**:
  - Marked `@pytest.mark.integration`; skipped by default; configurable via env (e.g., `OPENAI_MODEL`). Avoid hard-coded model names.
- **Config files**: add `pytest.ini` (markers, -q/-vv, filterwarnings) and coverage settings.

### Linting and typing
- Add `ruff` and `black`; enforce formatting and linting on CI.
- Add `mypy` with `python_version >= 3.11`, strict-ish settings; add `types-requests` for stubs.
- Optional: pre-commit hooks for `ruff`, `black`, `mypy`.

### CI/CD
- **CI (GitHub Actions)**:
  - Matrix: Python 3.11–3.13 on Ubuntu (and macOS if needed).
  - Steps: install (core only), ruff, black check, mypy, unit tests, coverage upload.
- **Release**:
  - On tag: build sdist+wheel (`python -m build`) and publish to PyPI using a token secret.
  - If using `hatch-vcs`, tags drive version; expose `__version__`.

### Backward compatibility & deprecation
- Start at `0.1.x`; move to `1.0.0` once API surface stabilizes.
- If a compat alias (`build_toolkit`) is kept, deprecate in docs and remove at `1.0`.
- Pin upper bounds on critical deps (`<0.4` for `langchain-core`, `<3` for `pydantic`) and revisit periodically.

### Security & robustness
- Document network behaviors and timeouts; recommend user-controlled retries (via wrappers) rather than internal implicit retries.
- Avoid auto network calls on import; keep `.env` loading guarded and documented.

### Immediate changes recommended for this repo
- Declare missing dependency: `pydantic>=2,<3` (used across loaders).
- Move `requests` under the `openapi` extra; remove `pytest`, `langgraph`, `langchain-openai`, `sentence-transformers`, `numpy`, `aiohttp` from core.
- Update README to use `build_tool_hub` and extras-based installs; remove outdated sections.
- Add `__version__` in `langtoolkit.__init__`.

### Release checklist
- [ ] All unit tests passing locally and on CI with coverage threshold (e.g., ≥90% for core modules).
- [ ] Lint (ruff), format (black), type-check (mypy) clean.
- [ ] README reflects current API and install variants; examples run (where applicable).
- [ ] Version bumped (or tag created if using VCS versioning).
- [ ] Build artifacts verified (sdist+wheel). Publish to TestPyPI first, then PyPI.

