from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from .tool import LoadedTool


def _build_args_schema_from_signature(
    func: Callable[..., Any],
    *,
    model_name: str,
    type_hint_fallback: type[Any] = str,
) -> type[BaseModel]:
    """Create a Pydantic schema model from a callable signature.

    If no type hints are present, falls back to `type_hint_fallback` (default: str).
    """
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        # Some builtins (e.g., many in math) don't expose signatures; use a generic schema
        return create_model(model_name, input=(type_hint_fallback, ...))  # type: ignore[return-value]
    fields: dict[str, tuple[type[Any], Any]] = {}
    for param_name, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # Skip *args/**kwargs in schema – LLMs struggle with those anyway
            continue

        annotated_type: type[Any]
        if param.annotation is not inspect._empty:  # type: ignore[attr-defined]
            annotated_type = param.annotation  # type: ignore[assignment]
        else:
            annotated_type = type_hint_fallback

        default_value = ... if param.default is inspect._empty else param.default
        fields[param_name] = (annotated_type, default_value)

    # Ensure at least one dummy field so the schema is valid
    if not fields:
        fields = {"input": (type_hint_fallback, ...)}

    return create_model(model_name, **fields)  # type: ignore[return-value]


class _CallableTool(BaseTool):
    """Generic wrapper that exposes a Python callable as a LangChain tool.

    Arguments are validated by a dynamically generated Pydantic model
    inferred from the target function's signature.
    """

    target: Callable[..., Any]
    args_schema: type[BaseModel]

    def _run(self, **kwargs: Any) -> Any:
        return self.target(**kwargs)

    async def _arun(self, **kwargs: Any) -> Any:
        # Avoid blocking the running event loop on sync callables
        result = self.target(**kwargs)
        if inspect.isawaitable(result):
            return await result
        try:
            import asyncio as _asyncio

            loop = _asyncio.get_running_loop()
            if loop and loop.is_running():
                return await _asyncio.to_thread(self.target, **kwargs)
        except RuntimeError:
            # No running loop
            pass
        return result


class SDKLoader:
    """Load tools from Python modules (SDKs) by introspecting callables.

    Example sources: "math", "os.path", custom installed SDK modules, etc.
    """

    def __init__(
        self,
        modules: Iterable[str | ModuleType | Any] | str | ModuleType | Any,
        *,
        include_predicate: Callable[[str, Callable[..., Any]], bool] | None = None,
        exclude_private: bool = True,
        max_tools_per_module: int | None = None,
    ) -> None:
        # Avoid using PEP 604 unions in isinstance which are not valid runtime checks
        if isinstance(modules, (str, ModuleType)) or not isinstance(modules, Iterable):
            self._sources: list[str | ModuleType | Any] = [modules]  # type: ignore[list-item]
        else:
            self._sources = list(modules)  # type: ignore[assignment]
        self._include_predicate = include_predicate
        self._exclude_private = exclude_private
        self._max_tools_per_module = max_tools_per_module

    @staticmethod
    def _sanitize_tool_name(raw_name: str) -> str:
        # OpenAI tool/function names must match ^[a-zA-Z0-9_-]+$
        return re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)

    def load(self) -> list[LoadedTool]:
        loaded: list[LoadedTool] = []
        used_names: set[str] = set()
        for source in self._sources:
            module: Any
            origin_name: str
            if isinstance(source, str):
                module = importlib.import_module(source)
                origin_name = module.__name__
            elif isinstance(source, ModuleType):
                module = source
                origin_name = module.__name__
            else:
                module = source
                cls = module.__class__
                # For object instances, use the class name only to keep tool names concise
                # and stable regardless of the defining module path. This also aligns with
                # common expectations like "SearXNGClient__search" in tests.
                origin_name = f"{cls.__name__}"
            candidates: list[tuple[str, Callable[..., Any]]] = []
            # For modules, inspect routines; for objects, include callables (methods allowed)
            predicate = (
                inspect.isroutine if isinstance(module, ModuleType) else callable
            )
            for attr_name, obj in inspect.getmembers(module, predicate):
                if self._exclude_private and attr_name.startswith("_"):
                    continue
                if self._include_predicate and not self._include_predicate(
                    attr_name, obj
                ):
                    continue
                # For module scanning, skip bound methods (shouldn't appear); for objects, allow methods
                if isinstance(module, ModuleType) and inspect.ismethod(obj):
                    continue
                candidates.append((attr_name, obj))

            # Stable order and optional truncation
            candidates.sort(key=lambda t: t[0])
            if self._max_tools_per_module is not None:
                candidates = candidates[: self._max_tools_per_module]

            for func_name, func in candidates:
                description = (
                    inspect.getdoc(func) or f"Function {func_name} from {origin_name}"
                )
                schema = _build_args_schema_from_signature(
                    func, model_name=f"{origin_name.replace('.', '_')}_{func_name}_Args"
                )
                base_name = self._sanitize_tool_name(f"{origin_name}__{func_name}")
                unique_name = base_name
                suffix_index = 2
                while unique_name in used_names:
                    unique_name = f"{base_name}_{suffix_index}"
                    suffix_index += 1
                used_names.add(unique_name)

                tool = _CallableTool(
                    name=unique_name,
                    description=description,
                    target=func,
                    args_schema=schema,
                )
                loaded.append(
                    LoadedTool(
                        name=tool.name,
                        description=description,
                        tool=tool,
                        source="sdk",
                        origin=origin_name,
                    )
                )

        return loaded
