"""Command-line entry point: ``python -m call_me_maybe``."""

import argparse
import json
import sys
import time
from pathlib import Path

from .pipeline import OutputRecord, run_function_calling
from .schemas import InputError, load_functions, load_prompts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='call_me_maybe',
        description='Translate natural-language prompts into function '
                    'calls using constrained decoding on Qwen3-0.6B.')
    parser.add_argument('--functions_definition', '-f', type=Path,
                        default=Path('data/input/functions_definition.json'),
                        help='JSON list of available functions')
    parser.add_argument('--input', '-i', type=Path,
                        default=Path('data/input/function_calling_tests.json'),
                        help='JSON list of prompts')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('data/output/'
                                     'function_calling_results.json'),
                        help='where to write the results')
    return parser.parse_args(argv)


def write_results(path: Path, results: list[OutputRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prompts = load_prompts(args.input)
        functions, raw_functions = load_functions(args.functions_definition)
    except InputError as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1

    start = time.monotonic()
    results = run_function_calling(prompts, functions, raw_functions)
    try:
        write_results(args.output, results)
    except OSError as e:
        print(f'\nError: cannot write {args.output}: {e}', file=sys.stderr)
        return 1
    minutes = (time.monotonic() - start) / 60
    print(f'\nWrote {len(results)} result(s) to {args.output} '
          f'in {minutes:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
