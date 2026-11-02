"""Pydantic models for the input files, and loaders that validate them."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

ParamType = Literal['string', 'number', 'boolean', None]
JsonDict = dict[str, Any]


class InputError(Exception):
    """Raised when an input file is missing or invalid."""


class ParamSpec(BaseModel):
    """Type of a function parameter or return value."""

    param_type: ParamType = Field(alias='type')


class FunctionDef(BaseModel):
    """A function the model may choose to call."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParamSpec]
    returns: ParamSpec


class PromptCase(BaseModel):
    """A natural-language request to turn into a function call."""

    prompt: str = Field(min_length=1)


def _load_json_list(path: Path, what: str) -> list[JsonDict]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InputError(f'{path}: {e}') from e
    if not isinstance(data, list) or not all(isinstance(item, dict)
                                             for item in data):
        raise InputError(f'{path}: {what} JSON must be a list of objects')
    return data


def load_functions(path: Path) -> tuple[list[FunctionDef], list[JsonDict]]:
    """Load and validate the function definitions.

    Returns:
        The validated functions, and the raw JSON objects (used verbatim
        in the model prompt).
    """
    raw = _load_json_list(path, 'functions')
    try:
        functions = [FunctionDef.model_validate(item) for item in raw]
    except ValidationError as e:
        raise InputError(f'{path}: {e}') from e
    if not functions:
        raise InputError(f'{path}: at least one function is required')
    names = [function.name for function in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InputError(f'{path}: duplicate function name(s): '
                         f'{", ".join(duplicates)}')
    return functions, raw


def load_prompts(path: Path) -> list[PromptCase]:
    """Load and validate the prompts to process."""
    raw = _load_json_list(path, 'prompts')
    try:
        return [PromptCase.model_validate(item) for item in raw]
    except ValidationError as e:
        raise InputError(f'{path}: {e}') from e
