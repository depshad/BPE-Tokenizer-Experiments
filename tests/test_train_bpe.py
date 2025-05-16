#!/usr/bin/env python3
import json
import time
from tokenizer.train import run_train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]

    # Debug merges
    merges_not_in_reference = [m for m in merges if m not in reference_merges]
    reference_not_in_merges = [m for m in reference_merges if m not in merges]

    print(f"Merges not in reference: {len(merges_not_in_reference)}")
    print(f"Reference merges not in learned merges: {len(reference_not_in_merges)}")
    print("Sample of merges not in reference:", merges_not_in_reference[:5])
    print("Sample of reference merges not in learned merges:", reference_not_in_merges[:5])


    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes(
                [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
            )
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }

    # Debug vocab
    vocab_keys_not_in_reference = set(vocab.keys()) - set(reference_vocab.keys())
    reference_keys_not_in_vocab = set(reference_vocab.keys()) - set(vocab.keys())
    vocab_values_not_in_reference = set(vocab.values()) - set(reference_vocab.values())
    reference_values_not_in_vocab = set(reference_vocab.values()) - set(vocab.values())

    print(f"Vocab keys not in reference: {len(vocab_keys_not_in_reference)}")
    print(f"Reference keys not in vocab: {len(reference_keys_not_in_vocab)}")
    print(f"Vocab values not in reference: {len(vocab_values_not_in_reference)}")
    print(f"Reference values not in vocab: {len(reference_values_not_in_vocab)}")
    print("Sample of vocab keys not in reference:", list(vocab_keys_not_in_reference)[:5])
    print("Sample of reference keys not in vocab:", list(reference_keys_not_in_vocab)[:5])
    print("Sample of vocab values not in reference:", list(vocab_values_not_in_reference)[:5])
    print("Sample of reference values not in vocab:", list(reference_values_not_in_vocab)[:5])

    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert merges == reference_merges

    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


