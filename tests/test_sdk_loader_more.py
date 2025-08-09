import os
import sys
from typing import Any
from unittest.mock import patch
import asyncio

ROOT = os.path.dirname(os.path.dirname(__file__))
PKG = os.path.join(ROOT, "langtoolkit")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from pydantic import BaseModel

from langtoolkit.sdk_loader import (
    SDKLoader,
    _build_args_schema_from_signature,
)
from langtoolkit.sdk_loader import _CallableTool  # type: ignore


def test_schema_varargs_skipped_and_dummy_added_when_empty():
    def only_varargs(*args, **kwargs):
        return None

    Model = _build_args_schema_from_signature(only_varargs, model_name="M")
    assert hasattr(Model, "model_fields") and "input" in Model.model_fields

    def func(a: int, *args, **kwargs):
        return a

    Model2 = _build_args_schema_from_signature(func, model_name="M2")
    fields = set(Model2.model_fields.keys())
    assert "a" in fields and "args" not in fields and "kwargs" not in fields


def test_schema_signature_failure_fallback(monkeypatch):
    def f(x):
        return x

    with patch("langtoolkit.sdk_loader.inspect.signature", side_effect=ValueError("x")):
        Model = _build_args_schema_from_signature(f, model_name="MF")
        assert "input" in Model.model_fields


def test_callable_tool_arun_awaitable_and_value_branches():
    async def aplus(x: int) -> int:
        return x + 1

    tool1 = _CallableTool(name="aplus", description="", target=aplus, args_schema=BaseModel)  # type: ignore[arg-type]

    async def run1():
        res = await tool1._arun(x=2)
        assert res == 3

    asyncio.run(run1())


