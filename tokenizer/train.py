from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, DefaultDict
from dataclasses import dataclass, field
import logging

from tokenizer.utils import pre_tokenize

logger = logging.getLogger(__name__)


@dataclass
class TokenNode:
    """
    Represents a node in a doubly linked list.
    Args:
    token_id (int): The token stored in this node.
    The token is integer ID from the vocabulary.
    - Initially, this may correspond to a raw byte value (0–255).
    - Later, it represents merged tokens with IDs >= 256.
    """
    token_id: int
    prev: Optional['TokenNode'] = field(default=None, repr=False)
    next: Optional['TokenNode'] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"ListNode({self.token_id})"


class TokenPairTracker:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # node pairs where a token pair occurs in a specific token ID context
        self.pair_occurrences_by_token_id: DefaultDict[
            Tuple[int, int], DefaultDict[int, List[Tuple['TokenNode', 'TokenNode']]]] = defaultdict(
            lambda: defaultdict(list))
        # Map from token pair to its frequency
        self.pair_frequencies: DefaultDict[Tuple[int, int], int] = defaultdict(int)

    def add_pair(self, token_pair: Tuple[int, int], token_id: int, token_node_pair: Tuple['TokenNode', 'TokenNode'], frequency: int):
        self.logger.debug(f"Adding token pair {token_pair} in token_id {token_id} with frequency {frequency}.")
        self.pair_occurrences_by_token_id[token_pair][token_id].append(token_node_pair)
        self.pair_frequencies[token_pair] += frequency

    def remove_pair(self, token_pair: Tuple[int, int], token_id: int, token_node_pair: Tuple['TokenNode', 'TokenNode'], frequency: int) -> None:
        self.logger.debug(f"Removing token pair {token_pair} from token_id {token_id} with frequency {frequency}.")
        self._remove_pair_occurrence(token_pair, token_id, token_node_pair)
        self._decrease_pair_frequency(token_pair, frequency)

    def _remove_pair_occurrence(self, token_pair: Tuple[int, int], token_id: int, token_node_pair: Tuple['TokenNode', 'TokenNode']) -> None:
        node_pairs = self.pair_occurrences_by_token_id[token_pair][token_id]
        if token_node_pair in node_pairs:
            node_pairs.remove(token_node_pair)
            if not node_pairs:
                del self.pair_occurrences_by_token_id[token_pair][token_id]
                if not self.pair_occurrences_by_token_id[token_pair]:
                    del self.pair_occurrences_by_token_id[token_pair]

    def _decrease_pair_frequency(self, token_pair: Tuple[int, int], frequency: int) -> None:
        if token_pair in self.pair_frequencies:
            self.pair_frequencies[token_pair] -= frequency
            if self.pair_frequencies[token_pair] <= 0:
                del self.pair_frequencies[token_pair]
                self.logger.debug(f"Token pair {token_pair} frequency dropped to 0 and was removed.")

    def get_most_frequent_pair(self, vocab: Dict[int, bytes]) -> Optional[Tuple[int, int]]:
        """
            Returns the most frequent token pair tracked during BPE training.
            In the case of a frequency tie (i.e., multiple token pairs with equal max frequency),
            the tie is broken **lexicographically by the actual merged byte content** of the token pair.
            This ensures consistency with GPT-2's BPE implementation, which prefers
            lexicographically last pairs when frequencies are equal.
            Args:
                vocab (Dict[int, bytes]):
                    Mapping of token ID to its corresponding byte sequence.
                    Used to compute the merged byte string of a pair for lexicographic comparison.
            Returns:
                Optional[Tuple[int, int]]: The token pair (as a tuple of token IDs) with the
                highest frequency. Returns `None` if no pairs are tracked.

            Example:
                Suppose the following token pairs and frequencies exist:
                    (105, 114) → b'ir' → freq = 10
                    (99, 111)  → b'co' → freq = 10
                    (116, 104) → b'th' → freq = 10

                Then the selected pair will be the one whose merged bytes are last
                in lexicographic order among ties:
                    b'th' > b'ir' > b'co' → selected = (116, 104)
            """
        if not self.pair_frequencies:
            return None

        def sort_key(pair: Tuple[int, int]):
            merged_bytes = vocab[pair[0]] + vocab[pair[1]]
            return self.pair_frequencies[pair], merged_bytes

        return max(self.pair_frequencies, key=sort_key)

    def has_pairs(self) -> bool:
        return bool(self.pair_frequencies)

class TokenSequenceBuilder:
    @staticmethod
    def build(
        byte_token_id_frequencies: Dict[int, int],
        id_to_token_bytes: Dict[int, Tuple[int, ...]]
    ) -> Tuple[Dict[int, TokenNode], TokenPairTracker]:
        """
        Builds linked list representations of each token and initializes the pair tracker.

        Args:
            byte_token_id_frequencies: token_id → frequency count
            id_to_token_bytes: token_id → byte sequence (tuple of ints)

        Returns:
            token_id_to_list_head: token_id → head TokenNode of linked list
            pair_tracker: initialized TokenPairTracker with pair counts
        """
        logger = logging.getLogger("TokenSequenceBuilder")
        token_id_to_list_head: Dict[int, TokenNode] = {}
        pair_tracker = TokenPairTracker()

        for token_id, freq in byte_token_id_frequencies.items():
            byte_sequence = id_to_token_bytes.get(token_id)
            if not byte_sequence:
                logger.warning(f"Token ID {token_id} has empty byte sequence. Skipping.")
                continue

            head = TokenNode(token_id=byte_sequence[0])
            current = head
            token_id_to_list_head[token_id] = head

            for byte in byte_sequence[1:]:
                new_node = TokenNode(token_id=byte)
                current.next = new_node
                new_node.prev = current

                pair = (current.token_id, new_node.token_id)
                pair_tracker.add_pair(pair, token_id, (current, new_node), freq)

                current = new_node

        return token_id_to_list_head, pair_tracker

@dataclass
class PreTokenization:
    """
    Represents the result of the pre-tokenization and byte-encoding phase
    of Byte Pair Encoding (BPE) training.

    This structure includes:
    - A mapping from byte-encoded tokens (tuples of ints) to assigned token IDs.
    - The reverse mapping from token IDs to byte-encoded tokens.
    - A list of token IDs in the order they appeared in the original corpus.
    - Frequency counts for each token ID.

    These byte-encoded tokens are derived from pre-tokenized input words using UTF-8 encoding.
    """
    token_bytes_to_id: Dict[Tuple[int, ...], int]
    id_to_token_bytes: Dict[int, Tuple[int, ...]]
    input_token_ids: List[int]
    byte_token_id_frequencies: Dict[int, int]  # token_id → frequency

@dataclass
class BPEModel:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]
    pair_to_merged_id: Dict[Tuple[int, int], int]

class MergeApplier:
    @staticmethod
    def apply_merge_to_sequences( new_token_id: int,
                                  token_pair_to_merge: Tuple[int, int],
                                  pair_tracker: 'TokenPairTracker',
                                  byte_token_id_frequencies: Dict[int, int],
                                  token_id_to_list_head: Dict[int, TokenNode]) -> None:
        """
            Applies a BPE merge by replacing all occurrences of a given token pair with a new merged token ID.

            This mutates the linked lists in-place by:
            - Creating new TokenNode instances for the merged token.
            - Rewiring prev/next pointers around the merged nodes.
            - Updating adjacent token pairs in the pair tracker.
            - Deleting the merged pair from the tracker.

            Args:
                new_token_id (int): The token ID assigned to the newly merged token.
                token_pair_to_merge (Tuple[int, int]): The token pair (e.g., (101, 116)) to be merged.
                pair_tracker (TokenPairTracker): Structure tracking all active token pairs.
                byte_token_id_frequencies (Dict[int, int]): Frequency count for each token ID.
                token_id_to_list_head (Dict[int, TokenNode]): Maps token IDs to the head node of their linked list.
        """
        if token_pair_to_merge not in pair_tracker.pair_occurrences_by_token_id:
            return  # No such pair — nothing to merge

        for token_id, node_pairs in pair_tracker.pair_occurrences_by_token_id[token_pair_to_merge].items():
            for left_node, right_node in node_pairs:
                # Create the new merged node
                new_node = TokenNode(new_token_id)
                new_node.prev, new_node.next = left_node.prev, right_node.next
                if left_node.prev:
                    left_node.prev.next = new_node
                else:
                    # This was the head node — update list head reference
                    token_id_to_list_head[token_id] = new_node

                if right_node.next:
                    right_node.next.prev = new_node

                frequency = byte_token_id_frequencies[token_id]

                MergeApplier.update_left_adjacent_pair(pair_tracker, frequency, token_id, new_node, left_node)
                MergeApplier.update_right_adjacent_pair(pair_tracker, frequency, token_id, new_node, right_node)


    def update_left_adjacent_pair(pair_tracker: 'TokenPairTracker', frequency: int, token_id: int,
                                   new_node: TokenNode, original_left: TokenNode):
        """
            Updates the pair on the left side of a merged node.

            Removes the old pair (prev, left) and inserts new pair (prev, merged).
        """
        if new_node.prev:
            old_pair = (new_node.prev.token_id, original_left.token_id)
            new_pair = (new_node.prev.token_id, new_node.token_id)

            # Example context before merge: (a, b, c, d, e)
            # If (b, c) is the pair being merged into (bc), then after merge: (a, bc, d, e)
            # This affects the adjacent pair on the left: (a, b)
            #   → Old adjacent pair (a, b) becomes (a, bc)
            #   → So we need to:
            #     - Remove (a, b) from the pair tracker
            #     - Add (a, bc) as a new pair

            # Remove the old left-adjacent pair involving the original left node
            pair_tracker.remove_pair(old_pair, token_id, (new_node.prev, original_left), frequency)

            # Add  new left-adjacent pair (a, bc) involving the newly merged node
            pair_tracker.add_pair(new_pair, token_id, (new_node.prev, new_node), frequency)

    def update_right_adjacent_pair(
            pair_tracker: 'TokenPairTracker',
            frequency: int,
            token_id: int,
            new_node: TokenNode,
            original_right: TokenNode
    ) -> None:
        if new_node.next:
            old_pair = (original_right.token_id, new_node.next.token_id)
            new_pair = (new_node.token_id, new_node.next.token_id)

            # Example context before merge: (a, b, c, d, e)
            # If (b, c) is the pair being merged into (bc), then after merge: (a, bc, d, e)
            #
            # This affects the adjacent pair on the right: (c, d)
            #   → Old adjacent pair (c, d) becomes (bc, d)
            #   → So we need to:
            #     - Remove (c, d) from the pair tracker
            #     - Add (bc, d) as a new pair

            # Remove the old right-adjacent pair involving the original right node
            pair_tracker.remove_pair(old_pair, token_id, (original_right, new_node.next), frequency)

            # Add the new right-adjacent pair involving the newly merged node
            pair_tracker.add_pair(new_pair, token_id, (new_node, new_node.next), frequency)



class BPETrainer:

    def __init__(self, vocab_size: int, special_tokens: List[str], regex_tokenizer_pattern: str):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.regex_tokenizer_pattern = regex_tokenizer_pattern

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"BPETrainer initialized with vocab_size={vocab_size} and special_tokens={special_tokens}")

    def train(self, text: str) -> BPEModel:

        vocab = {i: bytes([i]) for i in range(256)}  # Initial vocab
        vocab_index = len(vocab)
        merges: List[Tuple[bytes, bytes]] = []
        pair_to_merged_id: Dict[Tuple[int, int], int] = {} # {(token_id_1, token_id_2): merged_token_id}

        num_special_tokens = len(self.special_tokens)

        # Step 1: Pre-tokenization
        raw_text_tokens = pre_tokenize(pattern=self.regex_tokenizer_pattern, text=text)
        self.logger.debug(f"Regex pre-tokenization produced {len(raw_text_tokens)} raw text tokens.")

        # Step 2: Encode pre-tokens as byte sequences, assign token IDs and count frequencies
        pre_token_data = self._encode_and_index_tokens(
            raw_text_tokens)
        self.logger.debug(f"Pre-tokenization produced {len(pre_token_data.token_bytes_to_id)} unique byte-encoded tokens.")


        # Step 3: Initialize sequences and PairManager
        linked_lists, pair_tracker = TokenSequenceBuilder.build(
            pre_token_data.byte_token_id_frequencies,
            pre_token_data.id_to_token_bytes
        )

        # Step 4: BPE Merging Loop — build vocabulary by merging most frequent token pairs

        # Stop if the vocabulary size is reached
        while vocab_index < self.vocab_size - num_special_tokens:

            # Stop if no pairs remain
            if not pair_tracker.has_pairs():
                break

            # Identify the most frequent pair
            most_frequent_pair = pair_tracker.get_most_frequent_pair(vocab=vocab)
            if most_frequent_pair is None:
                break

            self.logger.debug(f"Merging pair: {most_frequent_pair}")

            # Merge the two tokens to form a new token (concatenated bytes)
            merged_token_bytes = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]
            vocab[vocab_index] = merged_token_bytes

            MergeApplier.apply_merge_to_sequences(vocab_index, most_frequent_pair, pair_tracker, pre_token_data.byte_token_id_frequencies, linked_lists)

            # Explicitly clean up the merged pair here
            del pair_tracker.pair_occurrences_by_token_id[most_frequent_pair]
            del pair_tracker.pair_frequencies[most_frequent_pair]

            # Update merges_id and merges
            pair_to_merged_id[most_frequent_pair[0], most_frequent_pair[1]] = vocab_index
            merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))
            self.logger.debug(f"Created new token {merged_token_bytes} with ID {vocab_index}.")
            vocab_index += 1

            self.logger.debug(f"Next available token ID: {vocab_index}")

        self._add_special_tokens(vocab=vocab)
        return BPEModel(vocab, merges, pair_to_merged_id)


    def _encode_and_index_tokens(self, pre_bpe_tokens: List[str]) -> PreTokenization:
        """
        Converts pre-tokenized text tokens into byte-encoded form and builds index mappings.

        Each string token is:
        - UTF-8 encoded into a tuple of bytes (integers),
        - Assigned a unique token ID,
        - Counted for frequency,
        - And tracked in the order it appeared.

        This method represents the first key step in the BPE pipeline after text pre-tokenization.

        Args:
            pre_bpe_tokens (List[str]): A list of text tokens produced by regex-based pre-tokenization.

        Returns:
            PreTokenization: A dataclass containing:
                - token_bytes_to_id: Mapping from UTF-8 byte tokens to token IDs.
                - id_to_token_bytes: Reverse mapping from token IDs to byte sequences.
                - input_token_ids: Ordered list of token IDs as they appeared in the corpus.
                - token_id_frequencies: Frequency count of each token ID.
        """
        self.logger.debug("Encoding pre-tokenized tokens into bytes and assigning token IDs.")

        token_bytes_to_id, id_to_token_bytes = {}, {}
        input_token_ids = []

        for token in pre_bpe_tokens:
            # Convert to byte tuple using UTF-8
            byte_encoded_token = tuple(token.encode("utf-8"))
            if byte_encoded_token not in token_bytes_to_id:
                token_id = len(token_bytes_to_id)
                token_bytes_to_id[byte_encoded_token] = token_id
                id_to_token_bytes[token_id] = byte_encoded_token
            else:
                token_id = token_bytes_to_id[byte_encoded_token]
            input_token_ids.append(token_id)

        # Count token frequencies after pre-tokenization
        byte_token_id_frequencies = Counter(input_token_ids)

        return PreTokenization(token_bytes_to_id=token_bytes_to_id,
                               id_to_token_bytes=id_to_token_bytes,
                               input_token_ids = input_token_ids,
                               byte_token_id_frequencies=byte_token_id_frequencies)


    def _add_special_tokens(self, vocab: Dict[int, bytes]) -> None:
        """
        Adds special tokens (e.g., <PAD>, <EOS>) to the vocabulary,
        encoding each string into bytes using UTF-8.
        """
        for token in self.special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")


def run_train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, 'r') as f:
        text = f.read()
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    trainer = BPETrainer(vocab_size, special_tokens, regex_tokenizer_pattern=GPT2_PAT)
    model = trainer.train(text=text)
    return model.vocab, model.merges


if __name__ == '__main__':
    input_path = '../tests/FIXTURES/corpus.en'
    vocab, merges = run_train_bpe(input_path, 500, ['<PAD>'])
    print(vocab)

