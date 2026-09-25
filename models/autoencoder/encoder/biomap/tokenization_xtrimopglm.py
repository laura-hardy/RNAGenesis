"""Tokenization classes for xTrimoPGLM."""

import os
from typing import List, Optional, Union, Dict, Any
from torch import TensorType
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


def load_vocab_file(vocab_file: str) -> List[str]:
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [line.strip() for line in lines]


class xTrimoPGLMTokenizer(PreTrainedTokenizer):
    """
    Constructs a xTrimoPGLM tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        eos_token: str = "<eos>",
        model_max_length: int = 2048,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

        if additional_special_tokens is None:
            additional_special_tokens = ['tMASK', 'gMASK', 'sMASK', 'eod', 'sop', 'eop', '<cls>', 'UNK', '</s>'] 

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            model_max_length=model_max_length,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.unique_no_split_tokens = self.all_tokens

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return text.split()

    def get_vocab(self) -> dict:
        base_vocab = self._token_to_id.copy()
        base_vocab.update(self.added_tokens_encoder)
        return base_vocab

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.eos_token_id]  
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return token_ids_0
            else:
                return token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return token_ids_0 + sep + token_ids_1 + sep  # Multiple inputs always have an EOS token


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "tokenizer.model")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)

    def apply_chat_template(
        self, 
        query, 
        add_generation_prompt: bool = True, 
        tokenize: bool = True, 
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        add_special_tokens: bool = True,
        **kwargs,
) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding]:

        generation_prompt = "<gmask><sop><eos>"
        if isinstance(query, str):
            query = [query]
        prompt_query = []
        if add_generation_prompt:
            for each in query:
                assert isinstance(each, str)
                prompt_query.append(generation_prompt+each)
        else:
            prompt_query = query
        if tokenize:
            output = self.batch_encode_plus(
                prompt_query,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                is_split_into_words=True,
                add_special_tokens=False
            )
            if return_dict:
                return output
            else:
                return output["input_ids"]
        else:
            return prompt_query