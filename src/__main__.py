#!/usr/bin/env python3

import json
from typing import Any, List, Union
import time
import os
import argparse
import sys
from src.validation import FunctionDef, TestPrompt
from pydantic import ValidationError
from src.decoder import ArgValue, Decoder

JsonDict = dict[str, Any]
OutputRecord = dict[str, Union[str, dict[str, ArgValue]]]
ValidatedInput = tuple[List[JsonDict], List[JsonDict], List[FunctionDef]]


def bar(i: int, total: int, width: int = 30) -> None:
    """Print a loading bar.

    Args:
        i (int): current iteration
        total (int): total iterations expected
        width (int, optional): how wide the bar should be. Defaults to 30.
    """
    done = int(width * (i + 1) / total)
    pct = (i + 1) / total * 100
    print("\rOverheating the PC...: ",
          f"[{'█' * done}{'·' * (width - done)}] {i+1}/{total} ({pct:.0f}%)",
          end='', flush=True)


def get_cli() -> argparse.Namespace:
    """Receive command line arguments

    Returns:
        argparse.Namespace: set of arguments received
    """
    cli: argparse.ArgumentParser = argparse.ArgumentParser()
    cli.add_argument('--functions_definition', '-f',
                     default='data/input/functions_definition.json')
    cli.add_argument('--input',
                     default='data/input/function_calling_tests.json')
    cli.add_argument('--output',
                     default='data/output/function_calling_results.json')

    args: argparse.Namespace = cli.parse_args()
    return args


def validate(args: argparse.Namespace) -> ValidatedInput:
    """Validate input so that it is correct JSON and has all needed keys.

    Args:
        args (argparse.Namespace): CLI args

    Returns:
        ValidatedInput: the prompts, the raw functions and the
                        validated functions
    """
    try:
        with open(args.input) as f:
            prompts = json.load(f)
        [TestPrompt(**prompt) for prompt in prompts]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as e:
        if e.__class__.__name__ == 'TypeError':
            print('Prompt JSON must be a list', file=sys.stderr)
            sys.exit(1)
        print(f"{args.input} raised an error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.functions_definition) as f:
            funcs = json.load(f)
        func_val: List[FunctionDef] = [FunctionDef(**func) for func in funcs]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as e:
        if e.__class__.__name__ == 'TypeError':
            print('Functions JSON must be a list', file=sys.stderr)
            sys.exit(1)
        print(f"{args.functions_definition} raised an error: {e}",
              file=sys.stderr)
        sys.exit(1)

    return (prompts, funcs, func_val)


def generate_output(prompts: List[JsonDict],
                    system_prompt: str,
                    funcs: List[JsonDict],
                    func_val: List[FunctionDef],
                    funcs_prompt: List[JsonDict]) -> List[OutputRecord]:
    """Generate output across many prompts.

    Args:
        prompts (List[JsonDict]): prompts from input JSON
        system_prompt (str): system prompt
        funcs (List[JsonDict]): the available functions in dictionary form
        func_val (List[FunctionDef]): the available functions validated.
                                      Needed for the decoder.
        funcs_prompt (List[JsonDict]): the functions manipulated for the
                                       prompt.

    Returns:
        List[OutputRecord]: one dictionary per prompt, holding the reply
    """
    outputs: List[OutputRecord] = []
    dec: Decoder = Decoder(funcs=func_val)
    for i in range(len(prompts)):
        try:
            bar(i, len(prompts))
        except ZeroDivisionError:
            pass

        output: OutputRecord = {
            "prompt": f"{prompts[i]['prompt']}"
        }

        output_prompt = f'''{{
\t"prompt": "{prompts[i]['prompt']}",
\t"name": "'''

        base_tokens = (dec.llm.encode(system_prompt).tolist()[0] +
                       dec.llm.encode(json.dumps(funcs_prompt)).tolist()[0] +
                       dec.llm.encode(output_prompt).tolist()[0]
                       )

        chosen_func = dec.constraint_decode_func(base_tokens)
        output['name'] = chosen_func
        output_prompt += chosen_func + '",\n' + '\t"parameters": {'

        for func in funcs:
            if func['name'] == chosen_func:
                parameters = func['parameters']

        base_tokens = (dec.llm.encode(json.dumps(funcs_prompt)).tolist()[0] +
                       dec.llm.encode(output_prompt).tolist()[0])
        output['parameters'] = dec.constraint_decode_args(parameters,
                                                          base_tokens)
        outputs.append(output)
    return outputs


def main() -> None:
    """
    Main function of this program.
    """
    args: argparse.Namespace = get_cli()

    prompts, funcs, func_val = validate(args)

    system_prompt: str = """
    For your answer, give me the name of the function
    from the possible functions provided.
    """

    funcs_prompt: List[JsonDict] = [{'name': func['name'],
                                     'description': func['description'],
                                     'parameters': func['parameters']}
                                    for func in funcs]

    start = time.time()

    outputs: List[OutputRecord] = generate_output(prompts,
                                                  system_prompt,
                                                  funcs,
                                                  func_val,
                                                  funcs_prompt)

    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        with open(args.output, 'w') as f:
            json.dump(outputs, f, indent=2)
    except OSError as e:
        print(f'\nProblem with output file: {e}', file=sys.stderr)

    end = time.time()

    print(f"\nRuntime was {round((end - start) / 60, 1)}")


if __name__ == '__main__':
    main()
