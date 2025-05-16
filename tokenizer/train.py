from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, DefaultDict
import regex as re
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ListNode:
    """
    Represents a node in a doubly linked list.
    Args:
    token (int): The token stored in this node.
    """
    token: int
    prev: Optional['ListNode'] = field(default=None, repr=False)
    next: Optional['ListNode'] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"ListNode({self.token})"


class PairManager:
    def __init__(self):
        #  byte pair and token IDs mapped to list of node pairs (left_node, right_node)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pair_info_map: DefaultDict[
            Tuple[int, int], DefaultDict[int, List[Tuple['ListNode', 'ListNode']]]] = defaultdict(
            lambda: defaultdict(list))
        # Map from pair to its frequency
        self.pair_frequencies: DefaultDict[Tuple[int, int], int] = defaultdict(int)

    def add_pair(self, pair: Tuple[int, int], token_id: int, node_pair: Tuple['ListNode', 'ListNode'], frequency: int):
        self.logger.debug(f"Adding pair {pair} with token_id {token_id} and frequency {frequency}.")
        self.pair_info_map[pair][token_id].append(node_pair)
        self.pair_frequencies[pair] += frequency

    def remove_pair(self, pair: Tuple[int, int], token_id: int, node_pair: Tuple['ListNode', 'ListNode'],
                    frequency: int) -> None:
        self.logger.debug(f"Removing pair {pair} with token_id {token_id} and frequency {frequency}.")
        self._remove_pair_from_info_map(pair, token_id, node_pair)
        self._decrease_pair_frequency(pair, frequency)

    def _remove_pair_from_info_map(self, pair: Tuple[int, int], token_id: int,
                                   node_pair: Tuple['ListNode', 'ListNode']) -> None:
        node_pairs = self.pair_info_map[pair][token_id]
        if node_pair in node_pairs:
            node_pairs.remove(node_pair)
            if not node_pairs:
                del self.pair_info_map[pair][token_id]
                if not self.pair_info_map[pair]:
                    del self.pair_info_map[pair]

    def _decrease_pair_frequency(self, pair: Tuple[int, int], frequency: int) -> None:
        if pair in self.pair_frequencies:
            self.pair_frequencies[pair] -= frequency
            if self.pair_frequencies[pair] <= 0:
                del self.pair_frequencies[pair]
                self.logger.debug(
                    f"Pair {pair} frequency dropped to {self.pair_frequencies.get(pair, 0)} and was removed.")

    def get_most_frequent_pair(self):
        if not self.pair_frequencies:
            return None
        return max(self.pair_frequencies, key=lambda pair: (self.pair_frequencies[pair], pair))

    def has_pairs(self) -> bool:
        return bool(self.pair_frequencies)


@dataclass
class TokenDictionaries:
    token_to_id: Dict[Tuple[int, ...], int]
    id_to_token: Dict[int, Tuple[int, ...]]
    token_ids: List[int]
    token_frequencies: Counter


class BPETrainer:

    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, text: str, pre_tokenize_pattern:str, vocab_size: int, special_tokens: List[str]):
        self.text = text
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pre_tokenize_pattern= pre_tokenize_pattern
        self.vocab = {i: bytes([i]) for i in range(256)}  # Initialize with all bytes
        self.merges: List[Tuple[bytes, bytes]] = []
        self.dict_merges_id: dict[Tuple[int, int],int] = {} # map from pair ids to merged token ID
        self.num_special_tokens = len(special_tokens)
        self.next_index = len(self.vocab)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"BPETrainer initialized with vocab_size={vocab_size} and special_tokens={special_tokens}.")

    def train(self) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

        # Step 1: Pre-tokenization
        pre_bpe_tokens = self.pre_tokenize(pattern=self.pre_tokenize_pattern)
        self.logger.debug(f"Pre-tokenization resulted in {len(pre_bpe_tokens)} tokens.")

        # Step 2: Create token dictionaries
        token_dicts = self._create_token_dictionaries(
            pre_bpe_tokens)
        self.logger.debug(f"Token dictionaries created with {len(token_dicts.token_to_id)} unique tokens.")

        # Step 3: Initialize sequences and PairManager
        linked_lists, pair_manager = self._initialize_sequences_and_pair_manager(token_dicts.token_frequencies,
                                                                                 token_dicts.id_to_token)
        self.logger.debug("Linked lists and PairManager initialized.")

        # Step 4: BPE Merging Loop
        while self.next_index < self.vocab_size - self.num_special_tokens:
            # Check if any pairs are left
            if not pair_manager.has_pairs():
                break

            # Identify the most frequent pair
            max_pair = pair_manager.get_most_frequent_pair()
            if max_pair is None:
                break

            # Create the new token by merging the two parts of the most frequent pair
            self.logger.info(f"Merging pair: {max_pair}")
            new_token = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.vocab[self.next_index] = new_token

            self._merge_pair(self.next_index, max_pair, pair_manager, token_dicts.token_frequencies, linked_lists)
            # Update merges_id and merges
            self.dict_merges_id[max_pair[0], max_pair[1]] = self.next_index
            self.merges.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))
            self.logger.debug(f"Created new token {new_token} with ID {self.next_index}.")
            self.next_index += 1
            self.logger.debug(f"Next available token ID: {self.next_index}")

        self._add_special_tokens()
        return self.vocab, self.merges, self.dict_merges_id

    def pre_tokenize(self, pattern: str) -> List[str]:
        tokens = re.findall(pattern, self.text)
        return tokens

    def _create_token_dictionaries(self, pre_bpe_tokens: List[str]) -> TokenDictionaries:
        """
        Initializes dictionaries for token-to-ID and ID-to-token mappings and counts token frequencies.

        Args:
            pre_bpe_tokens (List[str]): A list of tokens to be processed.

        Returns:
            Tuple[
                Dict[Tuple[int, ...], int],  # Mapping from byte-encoded tokens to their token IDs
                Dict[int, Tuple[int, ...]],  # Mapping from token IDs to byte-encoded tokens
                List[int],  # List of token IDs corresponding to the pre-BPE tokens
                Counter  # Frequency count of each token ID
            ]
        """
        self.logger.debug("Creating token dictionaries.")
        token_to_id, id_to_token = {}, {}
        token_ids = []

        # Initial tokenization
        for token in pre_bpe_tokens:
            byte_encoded_token = tuple(token.encode("utf-8"))
            if byte_encoded_token not in token_to_id:
                token_id = len(token_to_id)
                token_to_id[byte_encoded_token] = token_id
                id_to_token[token_id] = byte_encoded_token
            else:
                token_id = token_to_id[byte_encoded_token]
            token_ids.append(token_id)

        # Count token frequencies after pre-tokenization
        token_frequencies = Counter(token_ids)

        return TokenDictionaries(token_to_id=token_to_id,
                                 id_to_token=id_to_token,
                                 token_ids=token_ids,
                                 token_frequencies=token_frequencies)

    def _initialize_sequences_and_pair_manager(self,
                                               token_frequencies: Dict[int, int],
                                               id_to_token: Dict[int, Tuple[int, ...]]
                                               ) -> Tuple[List[ListNode], 'PairManager']:
        """
        Builds linked lists and initializes the PairManager.

        Returns:
            Tuple[List[ListNode], PairManager]:
                - A list of linked lists representing the token sequences.
                - An instance of PairManager containing pair information and frequencies.
        """
        logging.info("Initializing sequences and PairManager.")
        linked_lists: List[ListNode] = []
        pair_manager = PairManager()

        for token_id, token_freq in token_frequencies.items():
            token_bytes = id_to_token[token_id]

            logging.debug(f"Processing token ID {token_id} with frequency {token_freq}.")
            if not token_bytes:
                logging.warning(f"Token ID {token_id} has no bytes. Skipping.")
                continue

            head = ListNode(token_bytes[0])
            current = head
            linked_lists.append(head)

            for byte in token_bytes[1:]:
                new_node = ListNode(byte)
                current.next = new_node
                new_node.prev = current

                # Add pair to PairManager
                pair = (current.token, new_node.token)
                pair_manager.add_pair(pair, token_id, (current, new_node), token_freq)

                current = new_node

        logging.info("Sequences and PairManager initialization complete.")
        return linked_lists, pair_manager

    def _merge_pair(self, next_index: int,
                    max_pair: Tuple[int, int],
                    pair_manager: 'PairManager',
                    token_frequencies: Dict[int, int],
                    linked_lists: List[ListNode]) -> None:
        """
        Merges the most frequent pair and updates pair_manager.

        Args:
            next_index (int): The new token ID.
            max_pair (Tuple[int, int]): The pair to merge.
            pair_manager (PairManager): The PairManager instance.
            token_frequencies (Dict[int, int]): Token frequencies.
            linked_lists (List[ListNode]): List of linked lists.
        """
        if max_pair not in pair_manager.pair_info_map:
            return  # No pairs to merge

        for token_id, node_pairs in pair_manager.pair_info_map[max_pair].items():
            for left_node, right_node in node_pairs:
                # Create the new merged node
                new_node = ListNode(next_index)
                new_node.prev, new_node.next = left_node.prev, right_node.next
                if left_node.prev:
                    left_node.prev.next = new_node
                else:
                    # Update the head of the linked list
                    linked_lists[token_id] = new_node

                if right_node.next:
                    right_node.next.prev = new_node

                # Update adjacent pairs
                self._update_adjacent_pairs_after_merge(pair_manager,
                                                        token_frequencies,
                                                        token_id,
                                                        left_node,
                                                        right_node,
                                                        new_node)

        # Remove the merged pair from PairManager since the pair now exist as a single token
        del pair_manager.pair_info_map[max_pair]
        del pair_manager.pair_frequencies[max_pair]

    def _update_adjacent_pairs_after_merge(self,
                                           pair_manager: 'PairManager',
                                           token_frequencies: Dict[int, int],
                                           token_id: int,
                                           left_node: ListNode,
                                           right_node: ListNode,
                                           new_node: ListNode) -> None:
        token_freq = token_frequencies[token_id]
        self._update_left_adjacent_pair(pair_manager, token_freq, token_id, new_node, left_node)
        self._update_right_adjacent_pair(pair_manager, token_freq, token_id, new_node, right_node)

    def _update_left_adjacent_pair(self, pair_manager: 'PairManager', token_freq: int, token_id: int,
                                   new_node: ListNode, left_node: ListNode):
        if new_node.prev:
            old_pair = (new_node.prev.token, left_node.token)
            new_pair = (new_node.prev.token, new_node.token)

            # Merge process: (a, b, c, d, e) -> (a, b, cd, e)
            # Remove old pair(b,c) and decrease freq on left adjacent side
            pair_manager.remove_pair(old_pair, token_id, (new_node.prev, left_node), token_freq)

            # Add new pair (b,cd) on left adjacent side
            pair_manager.add_pair(new_pair, token_id, (new_node.prev, new_node), token_freq)

    def _update_right_adjacent_pair(self, pair_manager: 'PairManager', token_freq: int, token_id: int,
                                    new_node: ListNode, right_node: ListNode):
        if new_node.next:
            old_pair = (right_node.token, new_node.next.token)
            new_pair = (new_node.token, new_node.next.token)
            # Remove old pair
            pair_manager.remove_pair(old_pair, token_id, (right_node, new_node.next), token_freq)
            # Add new pair
            pair_manager.add_pair(new_pair, token_id, (new_node, new_node.next), token_freq)

    def _add_special_tokens(self):
        """
        Adds special tokens to the vocabulary.
        """
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token
        return


def run_train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, 'r') as f:
        text = f.read()
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    trainer = BPETrainer(text, GPT2_PAT, vocab_size, special_tokens)
    vocab, merges, dict_merges_id = trainer.train()
    return vocab, merges


if __name__ == '__main__':
    input_path = '../tests/FIXTURES/corpus.en'
    vocab, merges = run_train_bpe(input_path, 500, ['<PAD>'])
    print(vocab)

