"""Builds the model prompts and turns each request into a function call."""

import json

from .decoder import ArgValue, Decoder
from .schemas import FunctionDef, JsonDict, PromptCase

OutputRecord = dict[str, str | dict[str, ArgValue]]

SYSTEM_PROMPT = ('\n    For your answer, give me the name of the function\n'
                 '    from the possible functions provided.\n    ')


def progress_bar(done: int, total: int, width: int = 30) -> None:
    """Draw a single-line progress bar on stdout."""
    filled = width * done // total
    print(f'\rOverheating the PC...:  [{"█" * filled}{"·" * (width - filled)}]'
          f' {done}/{total} ({done / total:.0%})', end='', flush=True)


def call_header(prompt: str) -> str:
    """Start of the JSON object the model is asked to complete.

    The prompt is inserted verbatim (not JSON-escaped): escaping quotes
    did not improve results and changes what the model sees.
    """
    return '{\n\t"prompt": "' + prompt + '",\n\t"name": "'


def run_function_calling(prompts: list[PromptCase],
                         functions: list[FunctionDef],
                         raw_functions: list[JsonDict]) -> list[OutputRecord]:
    """Pick a function and its arguments for every prompt.

    Args:
        prompts: the requests to process.
        functions: the validated function definitions.
        raw_functions: the same definitions as raw JSON, shown to the model.

    Returns:
        One ``{"prompt", "name", "parameters"}`` record per prompt.
    """
    decoder = Decoder(functions)
    by_name = {function.name: function for function in functions}
    schema_tokens = decoder.encode(json.dumps([
        {'name': f['name'], 'description': f['description'],
         'parameters': f['parameters']}
        for f in raw_functions
    ]))
    system_tokens = decoder.encode(SYSTEM_PROMPT)

    results: list[OutputRecord] = []
    for i, case in enumerate(prompts):
        progress_bar(i + 1, len(prompts))
        header = call_header(case.prompt)
        name = decoder.choose_function(
            system_tokens + schema_tokens + decoder.encode(header))

        header += name + '",\n\t"parameters": {'
        arguments = decoder.decode_arguments(
            by_name[name].parameters, schema_tokens + decoder.encode(header))
        results.append({'prompt': case.prompt, 'name': name,
                        'parameters': arguments})
    return results
