"""Generate a human-readable reference for configuration options."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Type, Union, get_args, get_origin

import json

from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

from .settings import BotSettings


def _type_repr(annotation: Any) -> str:
    """Return a human friendly representation of ``annotation``."""

    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return annotation.__name__
        return str(annotation)
    args = get_args(annotation)
    if origin in {list, tuple}:
        inner = ", ".join(_type_repr(arg) for arg in args)
        name = "List" if origin is list else "Tuple"
        return f"{name}[{inner}]"
    if origin is dict:
        key, value = args or (Any, Any)
        return f"Dict[{_type_repr(key)}, {_type_repr(value)}]"
    if origin is Union:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return f"Optional[{_type_repr(non_none[0])}]"
        return f"Union[{', '.join(_type_repr(arg) for arg in args)}]"
    try:
        name = origin.__name__
    except AttributeError:  # pragma: no cover - defensive
        name = str(origin)
    inner = ", ".join(_type_repr(arg) for arg in args)
    return f"{name}[{inner}]"


def _default_repr(field) -> str | None:
    if field.default is not PydanticUndefined:
        value = field.default
    elif field.default_factory is not None:
        try:
            value = field.default_factory()
        except TypeError:  # pragma: no cover - defensive
            return None
    else:
        return None
    if isinstance(value, BaseModel):
        return "see nested section"
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:  # pragma: no cover - best effort
            return str(value)
    return str(value)


def _render_model(model_cls: Type[BaseModel], level: int = 2) -> Iterable[str]:
    """Yield markdown lines describing ``model_cls``."""

    prefix = "#" * level
    for name, field in model_cls.model_fields.items():
        alias = field.alias or name
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield f"{prefix} {alias}"
            if field.description:
                yield f"{field.description}\n"
            yield from _render_model(annotation, level + 1)
            continue
        yield f"{prefix} {alias}"
        yield f"- Type: `{_type_repr(annotation)}`"
        default = _default_repr(field)
        if default is not None:
            yield f"- Default: `{default}`"
        if field.description:
            yield f"- Description: {field.description}"
        yield ""


def generate_reference(path: Path | str | None = None) -> Path:
    """Generate the configuration reference document."""

    output_path = Path(path) if path is not None else Path(__file__).with_name("REFERENCE.md")
    lines = ["# Configuration Reference", ""]
    lines.append("The settings below are loaded by ``crypto_bot.config.settings.BotSettings``.")
    lines.append(
        "Environment variables use the ``BOT_`` prefix and ``__`` for nested fields "
        "(e.g. ``BOT_RISK__MAX_POSITIONS``)."
    )
    lines.append("")
    lines.extend(_render_model(BotSettings))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":  # pragma: no cover - manual utility
    generate_reference()
