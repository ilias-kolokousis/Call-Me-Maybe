from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, model_validator, Field
from src.validation import FunctionDef
from typing import List, Union
import numpy as np
import numpy.typing as npt

ArgValue = Union[str, float, bool, None]


class Decoder(BaseModel):
    """Decode LLM output using constraint decoding for performance.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    funcs: list[FunctionDef]
    llm: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    func_names: list[str] = []
    id_to_string: dict[int, str] = {}
    string_to_id: dict[str, int] = {}
    arg_tokens: list[int] = []

    def constraint_decode_func(self, base_tokens: List[int]) -> str:
        """Constraint decode only for function names.

        Args:
            base_tokens (List[int]): the prompt tokens

        Returns:
            str: the function name
        """
        base_logits = self.llm.get_logits_from_input_ids(base_tokens)
        base_logprobs = get_softmax(base_logits)

        scores = []
        for func in self.func_names:
            ids = self.llm.encode(func).tolist()[0]
            tokens = base_tokens.copy()
            total_logprobs = base_logprobs[ids[0]]
            tokens.append(ids[0])
            for tid in ids[1:]:
                logits = self.llm.get_logits_from_input_ids(tokens)
                logprobs = get_softmax(logits)
                total_logprobs += logprobs[tid]
                tokens.append(tid)
            scores.append(total_logprobs/len(ids))
        func_scores = zip(self.func_names, scores)
        return max(func_scores, key=lambda i: i[1])[0]

    def _is_legal_number(self, acc: str, token: str) -> bool:
        """Find out if candidates are a legal number.
        Example:
            "-45": legal
            "--45.4.2": illegal

        Args:
            acc (str): accumulated LLM output
            token (str): next possible token

        Returns:
            bool: true if legal, false if illegal
        """
        if len(token) <= 0:
            return False
        output: str = acc + token
        for char in token:
            if char not in '-0123456789.':
                return False
        minus_count: int = output.count('-')
        dot_count: int = output.count('.')
        if minus_count > 1 or dot_count > 1:
            return False
        if '-' in output and output[0] != '-':
            return False
        if '.' in output and output[0] == '.':
            return False
        if output.startswith('-.'):
            return False
        return True

    def constraint_decode_args(self,
                               parameters: dict[str, dict[str, str]],
                               base_tokens: List[int]) -> dict[str, ArgValue]:
        """Constraint decode only for the arguments.

        Args:
            parameters (dict[str, dict[str, str]]): parameters defined in
                                         the function_definition.json
            base_tokens (List[int]): prompt tokens

        Returns:
            dict[str, ArgValue]: LLM output in dictionary form
        """
        tokens = base_tokens.copy()
        output: dict[str, ArgValue] = {}
        for key, value in parameters.items():
            if value['type'] == 'string':
                tokens += self.llm.encode(f'"{key}": "').tolist()[0]
            else:
                tokens += self.llm.encode(f'"{key}": ').tolist()[0]
            acc = ''
            while len(acc) < 2000:
                logits = self.llm.get_logits_from_input_ids(tokens)

                if value['type'] == 'number':
                    cands = [i for i in self.arg_tokens
                             if self._is_legal_number(acc,
                                                      self.id_to_string[i])]
                    if acc and not acc.endswith(('-', '.')):
                        cands.append(self.string_to_id[','])
                    pick = max(cands, key=lambda i: logits[i])
                    if pick == self.string_to_id[',']:
                        tokens.append(pick)
                        tokens.append(self.string_to_id[' '])
                        try:
                            output[key] = float(acc)
                        except ValueError:
                            output[key] = None
                        break

                elif value['type'] == 'string':
                    cands = [i for i in self.id_to_string]
                    if acc:
                        cands.append(self.string_to_id['"'])
                    pick = max(cands, key=lambda i: logits[i])
                    if '"' in self.id_to_string[pick]:
                        tokens.append(pick)
                        tokens.append(self.string_to_id[' '])
                        output[key] = (acc
                                       + self.id_to_string[pick]
                                       ).strip(' ",}\n')
                        break

                elif value['type'] == 'boolean':
                    cands = [self.string_to_id['true'],
                             self.string_to_id['false']]
                    pick = max(cands, key=lambda i: logits[i])
                    tokens.append(pick)
                    tokens.append(self.string_to_id[','])
                    output[key] = self.id_to_string[pick] == 'true'
                    break

                else:
                    output[key] = None
                    break

                acc += self.id_to_string[pick]
                tokens.append(pick)
        return output

    @model_validator(mode='after')
    def build_vocab(self) -> "Decoder":
        """Build needed attributes
        """
        vocab_size: int = len(self.llm.get_logits_from_input_ids([0]))
        self.func_names: List[str] = [f.name for f in self.funcs]
        self.id_to_string: dict[int, str] = {x: self.llm.decode([x])
                                             for x in range(vocab_size)}
        self.string_to_id: dict[str, int] = {x[1]: x[0] for x in
                                             list(self.id_to_string.items())}
        self.arg_tokens: List[int] = []
        for id, string in self.id_to_string.items():
            if self._is_legal_number('', string) or string == '.':
                self.arg_tokens.append(id)

        return self


def get_softmax(logits: List[float]) -> npt.NDArray[np.float64]:
    """Get log-softmax from logits

    Args:
        logits (List[float]): logits

    Returns:
        npt.NDArray[np.float64]: log-softmax
    """
    x: npt.NDArray[np.float64] = np.asarray(logits, dtype=np.float64)
    x_max = float(x.max())
    log_sum_exp = float(np.log(np.exp(x - x_max).sum()))
    return x - (x_max + log_sum_exp)
