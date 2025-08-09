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
    def _infer_type_and_default(param: inspect.Parameter) -> tuple[type[Any], Any]:
        # Prefer explicit annotation
        if param.annotation is not inspect._empty:  # type: ignore[attr-defined]
            annotated_type: type[Any] = param.annotation  # type: ignore[assignment]
            default_value = ... if param.default is inspect._empty else param.default
            return annotated_type, default_value
        # Infer from default value when available
        if param.default is not inspect._empty:
            dv = param.default
            if isinstance(dv, bool):
                return bool, dv
            if isinstance(dv, int):
                return int, dv
            if isinstance(dv, float):
                return float, dv
            if isinstance(dv, str):
                return str, dv
            if isinstance(dv, list):
                return list, dv
            if isinstance(dv, dict):
                return dict, dv
            # Unknown/sentinel defaults (e.g., object()) – accept anything, default None
            return Any, None
        # Fallback
        return type_hint_fallback, ...

    for param_name, param in signature.parameters.items():
        # Never expose implicit receiver parameters
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # Skip *args/**kwargs in schema – LLMs struggle with those anyway
            continue

        annotated_type, default_value = _infer_type_and_default(param)
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
                # Distinguish between class objects and instances for clearer naming.
                # - For class objects (e.g., yfinance.Ticker), use the top-level package
                #   name (e.g., "yfinance") so tool names start with the SDK package.
                # - For instances, keep using the class name to produce concise names
                #   like "SearXNGClient__search" as expected by tests.
                if inspect.isclass(module):
                    mod_name = getattr(module, "__module__", "") or ""
                    pkg_root = mod_name.split(".")[0] if mod_name else module.__name__
                    origin_name = pkg_root
                else:
                    cls = module.__class__
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
                # If the source is a class object and the callable is an instance method
                # (i.e., it expects a `self`), synthesize a wrapper that constructs the
                # instance from constructor args and then calls the method with its own args.
                if inspect.isclass(source):
                    try:
                        method_sig = inspect.signature(func)
                    except (TypeError, ValueError):
                        method_sig = None
                    is_instance_method = False
                    if method_sig is not None:
                        params = list(method_sig.parameters.values())
                        if params and params[0].name in ("self",):
                            is_instance_method = True
                    if is_instance_method:
                        # Build field maps from constructor and method (excluding self)
                        try:
                            ctor_sig = inspect.signature(source.__init__)  # type: ignore[attr-defined]
                        except (TypeError, ValueError):
                            ctor_sig = None

                        field_defs: dict[str, tuple[type[Any], Any]] = {}

                        def _infer_type_and_default(param: inspect.Parameter) -> tuple[type[Any], Any]:
                            if param.annotation is not inspect._empty:  # type: ignore[attr-defined]
                                annotated_type: type[Any] = param.annotation  # type: ignore[assignment]
                                default_value = ... if param.default is inspect._empty else param.default
                                return annotated_type, default_value
                            if param.default is not inspect._empty:
                                dv = param.default
                                if isinstance(dv, bool):
                                    return bool, dv
                                if isinstance(dv, int):
                                    return int, dv
                                if isinstance(dv, float):
                                    return float, dv
                                if isinstance(dv, str):
                                    return str, dv
                                if isinstance(dv, list):
                                    return list, dv
                                if isinstance(dv, dict):
                                    return dict, dv
                                return Any, None
                            return str, ...

                        def _add_fields_from(sig: inspect.Signature | None) -> list[str]:
                            names: list[str] = []
                            if sig is None:
                                return names
                            for p in sig.parameters.values():
                                if p.name in ("self", "cls"):
                                    continue
                                if p.kind in (
                                    inspect.Parameter.VAR_POSITIONAL,
                                    inspect.Parameter.VAR_KEYWORD,
                                ):
                                    continue
                                annotated_type, default_value = _infer_type_and_default(p)
                                if p.name not in field_defs:
                                    field_defs[p.name] = (annotated_type, default_value)
                                    names.append(p.name)
                            return names

                        ctor_param_names = _add_fields_from(ctor_sig)
                        # Method fields should be added after ctor fields to allow method params to override
                        method_param_names = _add_fields_from(method_sig)

                        # Create args schema from combined fields
                        model_name = (
                            f"{origin_name.replace('.', '_')}_{getattr(source, '__name__', 'Cls')}_{func_name}_Args"
                        )
                        schema = create_model(model_name, **field_defs)  # type: ignore[arg-type]

                        # Build the wrapper target
                        def _make_wrapper(_cls: Any, _method_name: str, _ctor_names: list[str], _method_names: list[str]):
                            def _wrapper(**kwargs: Any) -> Any:
                                ctor_kwargs = {k: kwargs[k] for k in _ctor_names if k in kwargs}
                                method_kwargs = {k: kwargs[k] for k in _method_names if k in kwargs}
                                instance = _cls(**ctor_kwargs)
                                bound_method = getattr(instance, _method_name)
                                return bound_method(**method_kwargs)

                            return _wrapper

                        target = _make_wrapper(source, func_name, ctor_param_names, [n for n in method_param_names if n not in ("self", "cls")])

                        description = (
                            inspect.getdoc(func)
                            or f"Method {getattr(source, '__name__', 'Class')}.{func_name} from {origin_name}"
                        )
                        base_name = self._sanitize_tool_name(
                            f"{origin_name}__{getattr(source, '__name__', 'Class')}__{func_name}"
                        )
                        unique_name = base_name
                        suffix_index = 2
                        while unique_name in used_names:
                            unique_name = f"{base_name}_{suffix_index}"
                            suffix_index += 1
                        used_names.add(unique_name)

                        tool = _CallableTool(
                            name=unique_name,
                            description=description,
                            target=target,
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
                        continue

                # Default path: functions, module-level callables, or instance callables
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
