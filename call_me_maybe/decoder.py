"""Constrained decoding on top of a small causal language model.

Instead of letting the model generate free text and hoping it is valid,
every decoding step only considers the tokens that keep the output valid
for the expected type (function name, number, string or boolean).
"""

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model

from .schemas import FunctionDef, ParamSpec

ArgValue = str | float | bool | None
FloatArray = npt.NDArray[np.float64]

# Upper bounds for a single argument value, so decoding always ends even
# if the model never produces a closing token.
MAX_VALUE_CHARS = 2000
MAX_VALUE_TOKENS = 512


def log_softmax(logits: list[float]) -> FloatArray:
    """Numerically stable log-softmax."""
    x: FloatArray = np.asarray(logits, dtype=np.float64)
    x_max = float(x.max())
    log_sum_exp = float(np.log(np.exp(x - x_max).sum()))
    return x - (x_max + log_sum_exp)


def is_legal_number(prefix: str, token: str) -> bool:
    """Whether ``prefix + token`` can still grow into a valid number.

    Examples: ``"-45"`` and ``"4."`` are legal, ``"--4"`` and ``"4.5.2"``
    are not.
    """
    if not token or any(char not in '-0123456789.' for char in token):
        return False
    text = prefix + token
    if text.count('-') > 1 or text.count('.') > 1:
        return False
    if '-' in text and not text.startswith('-'):
        return False
    return not text.startswith(('.', '-.'))


class Decoder:
    """Picks a function and decodes its arguments for a given prompt."""

    def __init__(self, functions: list[FunctionDef],
                 llm: Small_LLM_Model | None = None) -> None:
        self.llm = llm if llm is not None else Small_LLM_Model()
        self.function_names = [function.name for function in functions]
        self._name_tokens = {name: self.encode(name)
                             for name in self.function_names}

        vocab_size = len(self.llm.get_logits_from_input_ids([0]))
        self.id_to_string = [self.llm.decode([i]) for i in range(vocab_size)]
        # Several ids can decode to the same text; the last one wins.
        self.string_to_id = {text: i for i, text
                             in enumerate(self.id_to_string)}
        self.number_ids = [i for i, text in enumerate(self.id_to_string)
                           if is_legal_number('', text) or text == '.']
        self.comma = self.string_to_id[',']
        self.space = self.string_to_id[' ']
        self.true = self.string_to_id['true']
        self.false = self.string_to_id['false']

    def encode(self, text: str) -> list[int]:
        ids: list[int] = self.llm.encode(text).tolist()[0]
        return ids

    def choose_function(self, prompt_tokens: list[int]) -> str:
        """Return the function name the model finds most likely.

        Each name is scored by its average token log-probability, so long
        names are not penalised for having more tokens.
        """
        base_logprobs = log_softmax(
            self.llm.get_logits_from_input_ids(prompt_tokens))
        best_name, best_score = '', -np.inf
        for name in self.function_names:
            ids = self._name_tokens[name]
            tokens = prompt_tokens + ids[:1]
            total = base_logprobs[ids[0]]
            for token_id in ids[1:]:
                logprobs = log_softmax(
                    self.llm.get_logits_from_input_ids(tokens))
                total += logprobs[token_id]
                tokens.append(token_id)
            score = total / len(ids)
            if score > best_score:
                best_name, best_score = name, score
        return best_name

    def decode_arguments(self, parameters: dict[str, ParamSpec],
                         prompt_tokens: list[int]) -> dict[str, ArgValue]:
        """Decode a value for every parameter, in order."""
        tokens = prompt_tokens.copy()
        arguments: dict[str, ArgValue] = {}
        for name, spec in parameters.items():
            if spec.param_type == 'string':
                tokens += self.encode(f'"{name}": "')
                arguments[name] = self._decode_string(tokens)
            elif spec.param_type == 'number':
                tokens += self.encode(f'"{name}": ')
                arguments[name] = self._decode_number(tokens)
            elif spec.param_type == 'boolean':
                tokens += self.encode(f'"{name}": ')
                arguments[name] = self._decode_boolean(tokens)
            else:
                tokens += self.encode(f'"{name}": ')
                arguments[name] = None
        return arguments

    def _logits(self, tokens: list[int]) -> FloatArray:
        return np.asarray(self.llm.get_logits_from_input_ids(tokens),
                          dtype=np.float64)

    def _decode_number(self, tokens: list[int]) -> float | None:
        """Decode digits, '-' and '.' until the model emits a comma.

        Appends the generated tokens (and the closing ", ") to ``tokens``.
        """
        acc = ''
        for _ in range(MAX_VALUE_TOKENS):
            if len(acc) >= MAX_VALUE_CHARS:
                break
            logits = self._logits(tokens)
            candidates = [i for i in self.number_ids
                          if is_legal_number(acc, self.id_to_string[i])]
            if acc and not acc.endswith(('-', '.')):
                candidates.append(self.comma)
            pick = max(candidates, key=lambda i: logits[i])
            if pick == self.comma:
                tokens += [self.comma, self.space]
                break
            acc += self.id_to_string[pick]
            tokens.append(pick)
        try:
            return float(acc)
        except ValueError:
            return None

    def _decode_string(self, tokens: list[int]) -> str:
        """Decode freely until the model emits a token containing '"'.

        Appends the generated tokens (and a trailing space) to ``tokens``.
        """
        acc = ''
        for _ in range(MAX_VALUE_TOKENS):
            if len(acc) >= MAX_VALUE_CHARS:
                break
            pick = int(np.argmax(self._logits(tokens)))
            text = self.id_to_string[pick]
            if '"' in text:
                tokens += [pick, self.space]
                return (acc + text).strip(' ",}\n')
            acc += text
            tokens.append(pick)
        return acc.strip(' ",}\n')

    def _decode_boolean(self, tokens: list[int]) -> bool:
        logits = self._logits(tokens)
        pick = self.true if logits[self.true] >= logits[self.false] \
            else self.false
        tokens += [pick, self.comma]
        return pick == self.true
