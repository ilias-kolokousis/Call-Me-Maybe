_This project has been created as part of the 42 curriculum by ikolokou._

# Description
In this project, we need to build an LLM model using the Qwen/Qwen3-0.6B model as our foundation. The input is prompts and functions in JSON files. The model must understand from the prompt which is the most suitable function to use in each case and also figure out the suitable parameters.

# Instructions
1. If it's first time running this project, run `make install` to install all needed dependencies and create a virtual environment.
2. To run it, use `make run`. Warning this take about 4 minutes to run.
3. You can use the following flags for input in different directories with `uv run python -m src`: 
    - `--functions_definition`: for the function definitions file
    - `--input`: For the prompts JSON
    - `--output`: For the output JSON
3. For debugging mode, use `make debug`.
4. To clean any cache, like pycache or mypy_cache, use `make clean`.
5. To use flake8 and mypy, use `make lint`.
6. For a stricter version of mypy use `make lint-strict`. This is optional, however.

# Resources
- [Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)

_AI has been used mostly as a guide to help me when I'm stuck. It has been specifically instructed to guide me to the solution instead of offering it immediately. It also helped figure out any mypy errors._

# Algortihm Explanation 

The main idea of this project is to use **constraint decoding**. Even small model like Qwen, have a vocabulary of ~151K entries. If we approach this naively, then we would have to check each entry and find the most probable one to follow. That's expensive if an output could consist of even 50 tokens.

Instead, we limit the pool of usable entries. For example, if we know we are looking for numbers, then only digits and the signs '.' and '-' are legal. From ~151K we manage to drop to just 12 entries!

# Design Choices

Function names are a special case. To increase accuracy, I calculate the probability of all function names and normalize to the length of the function name. Short names are easier to get predicted. 

# Performance Analysis

This algorithm manages to achieve a 100% accuracy on the test prompts. It runs for about 3.7 minutes.

# Challenges Faced

Function names would generally work with some simple constrained decoding. However, `fn_substitute_string_with_regex` would get under-predicted. That is mostly because of the length. So, instead of letting the LLM choose one immediately, I actually calculate the score for each name in each prompt and pick the one with the highest probability normalized for the length of the name.

# Testing Strategy

I validated mostly on the test prompts. However, I changed the pormpts to more weird configurations to see how succesful the model is. I also tested on ambiguous prompts, like "What is the sum of the square root of 4 and square root of 3".

# Example Usage

For the following prompt:

```JSON
  {
    "prompt": "Replace all vowels in 'Programming is fun' with asterisks"
  }
```

the following functions were used:
```JSON
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {
        "type": "string"
      },
      "regex": {
        "type": "string"
      },
      "replacement": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
```

When ran, the command line shows a loading bar to inform how many prompts have been processed:
```bash
Overheating the PC...:  [█████·························] 2/11 (18%)
```

And at the end, it shows how long it took to run the prompts:
```bash
Runtime was 3.7
```

At the end, the user can find the output in `data/output/function_calling_results.json`. In this case, here is the output:
```JSON
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