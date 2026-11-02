_This project has been created as part of the 42 curriculum by ikolokou._

# Call Me Maybe

Function calling with a small language model, made reliable through
**constrained decoding**.

Given a list of function definitions and a natural-language request, Call Me
Maybe uses [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) to pick the
right function and fill in its arguments with correctly typed values. The model
never generates free text: at every decoding step, only the tokens that keep
the output valid are allowed.

```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Programming is fun",
    "regex": "([aeiouAEIOU])",
    "replacement": "*"
  }
}
```

## Getting started

Requirements: Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). The first
run downloads the Qwen3-0.6B weights (~1.5 GB) from Hugging Face.

```sh
make install   # create .venv and install dependencies
make run       # process data/input/function_calling_tests.json
```

Results are written to `data/output/function_calling_results.json`.

Custom inputs and outputs:

```sh
uv run python -m call_me_maybe \
    --functions_definition path/to/functions.json \
    --input path/to/prompts.json \
    --output path/to/results.json
```

Other targets: `make lint` (flake8 + mypy), `make lint-strict` (`mypy --strict`),
`make debug` (runs under pdb) and `make clean`.

## Input format

**Function definitions** (`--functions_definition`) are a JSON list of functions. Each parameter's type
is `string`, `number`, `boolean` or `null`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

**Prompts** (`--input`) are a JSON list of objects with a `prompt` key:

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

Both files are validated with pydantic before the model is loaded. Invalid
input (malformed JSON, missing keys, unknown types, duplicate function names)
is reported with a clear error and exit status `1`.

## How it works

### Constrained decoding

Even a small model like Qwen3-0.6B has a vocabulary of ~150K tokens. Letting
it generate freely and then parsing the result is fragile: it might invent a
function name, write `"five"` for a number, or never close a string. Instead,
each decoding step masks out every token that would make the output invalid,
and picks the most likely of the remaining ones:

| Value type | Allowed tokens |
|---|---|
| number | Tokens made of `0-9`, `-` and `.` that keep the text a valid number. `,` ends the value. |
| string | Any token. A token containing `"` ends the value. |
| boolean | Only `true` or `false`. |

For numbers this shrinks the candidate set from ~150K tokens to a few hundred.
Every value is also capped in length, so decoding always ends.

### Choosing the function

The model is shown a short instruction, the function definitions as JSON, and
the beginning of a JSON object to complete:

```
{
	"prompt": "What is the sum of 2 and 3?",
	"name": "
```

Rather than generating the name token by token, every known function name is
scored by its **average token log-probability**, and the best one wins.
Averaging matters: summing log-probabilities penalises long names such as
`fn_substitute_string_with_regex`, which were consistently under-predicted
before this change.

### Filling in the arguments

The chosen name and `"parameters": {` are appended to the context. Then, for
each parameter in order, the parameter's key is appended and its value is
decoded with the constraints above.

## Results

On the 11 prompts of the original 42 test set, the decoder chose the correct
function and arguments for every prompt (100%), taking about 3.7 minutes on a
laptop CPU. Most of that time goes to model forward passes: the provided SDK
re-runs the whole context for every generated token.

### Limitations

- A 0.6B model makes mistakes that the constraints cannot catch. For example,
  in "Add -4.5 and 10.25", the minus sign is sometimes dropped (`4.5`). The
  output is always well-formed, but not always right.
- The SDK has no KV cache, so every generated token re-runs the full context. This makes long string arguments slow.

I also tested ambiguous or unusual prompts (e.g. "What is the sum of the square
root of 4 and square root of 3"), to see how far the model can be pushed.

## Project structure

```
Call-Me-Maybe/
├── call_me_maybe/
│   ├── __main__.py     # command-line entry point
│   ├── schemas.py      # pydantic models and input loading
│   ├── pipeline.py     # prompt construction and the per-prompt loop
│   └── decoder.py      # constrained decoding on top of the LLM
├── llm_sdk/            # model wrapper provided by the 42 subject
├── data/input/         # example function definitions and prompts
├── Makefile
└── pyproject.toml
```

## Resources

- [Constrained decoding (Aidan Cooper)](https://www.aidancooper.co.uk/constrained-decoding/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

AI was used mostly as a guide when I was stuck. It was explicitly told to steer
me towards the solution rather than give it to me. It also helped me resolve
mypy errors.
