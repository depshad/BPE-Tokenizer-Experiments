import pytest
from tokenizer.tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    # Vocabulary (char -> id)
    inverse_vocab = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'i': 8,
        'n': 9,
        't': 10,
        'ea': 11,  # from merging 'e' (5) and 'a' (1)
        'eat': 12, # from merging 'ea' (11) and 't' (10) — now has a lower number than 'ti'
        'in': 13,  # from merging 'i' (8) and 'n' (9)
        'ti': 14,  # from merging 't' (10) and 'i' (8) — now has a higher number, so lower priority
        'ab': 15,
        'bc': 16,
        'aa': 17,
        'abc': 18,
        'bd': 19,
    }
    # Invert the mapping so that keys are token ids and values are the token strings
    vocab = {v: k for k, v in inverse_vocab.items()}
    # Merges as an ordered list of byte pairs
    merges = [
        (b'e', b'a'),
        (b'ea', b't'),
        (b'i', b'n'),
        (b't', b'i'),
        (b'a', b'b'),
        (b'b', b'c'),
        (b'a', b'a'),
        (b'ab', b'c'),
        (b'b', b'd'),
        ]
    # Dictionary for merges mapping from a tuple of token ids to the new merged token id
    # dict_merges_id = {(5, 1): 11, (11, 10): 12, (8, 9): 13, (10, 8): 14, (1, 2): 15, (2, 3): 16, (1, 1): 17, (15, 3): 18, (2, 4): 19}
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=None)