"""JSON serialization helpers for the decoupled IR contracts."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from . import schema as ir_schema


T = TypeVar("T")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in payload:
            continue
        kwargs[field.name] = _coerce_value(type_hints.get(field.name, field.type), payload[field.name])
    return cls(**kwargs)  # type: ignore[misc]


def read_dataclass_json(path: str | Path, cls: type[T]) -> T:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ir_schema.ContractError(f"Expected JSON object in {path}")
    return dataclass_from_dict(cls, payload)


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple):
        item_type = args[0] if args else Any
        return [_coerce_value(item_type, item) for item in value]
    if origin is dict:
        return dict(value)
    if origin in (Union, UnionType) and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)]
        return _coerce_value(non_none[0], value) if non_none else value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is ir_schema.BBox and isinstance(value, list):
        return ir_schema.BBox.from_list(value)
    if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
        return dataclass_from_dict(annotation, value)
    return value
