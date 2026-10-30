#!/usr/bin/env python3

from pydantic import BaseModel, Field
from typing import Literal


class ParamSpec(BaseModel):
    """Pydantic class for input validation only for function parameters
    """
    param_type: Literal['string', 'number', 'boolean', None] = Field(
        alias='type')


class FunctionDef(BaseModel):
    """Pydantic class for input validation only for functions
    """
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParamSpec]
    returns: ParamSpec


class TestPrompt(BaseModel):
    """Pydantic class for input validation only for prompts
    """
    prompt: str = Field(min_length=1)
