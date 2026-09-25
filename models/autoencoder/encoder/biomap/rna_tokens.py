"""Character-wise RNA token ids for the pinned xTrimoPGLM tokenizer.

Valid nonempty canonical A/C/G/U strings are passed unchanged to
`tokenizer.convert_tokens_to_ids(sequence)`. This wrapper does not strip,
uppercase, lowercase, rewrite T to U, accept ambiguity codes, or normalize
whitespace. Any other input raises ValueError before the pinned tokenizer
is called. Padding id 0 is the base-vocabulary index of `<pad>`, not
`convert_tokens_to_ids("<pad>")`.
"""

import os

from .tokenization_xtrimopglm import xTrimoPGLMTokenizer

_BIOMAP_DIR = os.path.dirname(os.path.abspath(__file__))
PAD_TOKEN_ID = 0
_CANONICAL_RNA = frozenset("ACGU")


def load_tokenizer():
    return xTrimoPGLMTokenizer(
        vocab_file=os.path.join(_BIOMAP_DIR, "tokenizer.model"),
    )


def encode_rna_characters(sequence, tokenizer=None):
    """Map each canonical RNA character to a vocab id.

    The input string is not rewritten. Non-str, empty, and noncanonical
    input raises ValueError and is not delegated to the pinned tokenizer.
    """
    if not isinstance(sequence, str) or not sequence or any(ch not in _CANONICAL_RNA for ch in sequence):
        raise ValueError(
            "sequence must be a nonempty str of canonical RNA characters A, C, G, U; "
            "got %r" % (sequence,)
        )
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.convert_tokens_to_ids(sequence)
