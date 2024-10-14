from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, DefaultDict
import regex as re
from dataclasses import dataclass, field
from typing import Optional


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


class BPETrainer:
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, text: str, vocab_size: int, special_tokens: List[str]):
        self.text = text
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {i: bytes([i]) for i in range(256)}  # Initialize with all bytes
        self.merges: List[Tuple[bytes, bytes]] = []
        self.num_special_tokens = len(special_tokens)
        self.next_index = len(self.vocab)

    def train(self) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

        # Initial tokenization
        pre_bpe_tokens = self.pre_tokenize()

        # Initialize dictionaries and token ID list
        dict_token_to_id, dict_id_to_token, token_ids, token_frequencies = self._create_token_dictionaries(
            pre_bpe_tokens)

        linked_lists, pair_info_map, pair_frequencies = self._build_linked_lists_and_pair_info(token_frequencies,
                                                                                               dict_id_to_token)

        while self.next_index < self.vocab_size - self.num_special_tokens:
            # Identify the most frequent pair
            max_pair = max(pair_frequencies, key=lambda pair: (pair_frequencies[pair], pair))
            # Create the new token by merging the two parts of the most frequent pair
            new_token = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.vocab[self.next_index] = new_token
            self._merge_pair(self.next_index, max_pair, pair_info_map,
                             pair_frequencies, token_frequencies,
                             linked_lists)
            self.merges.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))

            self.next_index += 1

        self._add_special_tokens()
        return self.vocab, self.merges

    def pre_tokenize(self, pattern: str = GPT2_PAT) -> List[str]:
        tokens = re.findall(pattern, self.text)
        return tokens

    def _create_token_dictionaries(self, pre_bpe_tokens: List[str]) -> Tuple[Dict[Tuple[int, ...], int],
                                                                             Dict[int, Tuple[int, ...]],
                                                                             List[int], Counter]:
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
        dict_token_to_id: Dict[Tuple[int, ...], int] = {}
        dict_id_to_token: Dict[int, Tuple[int, ...]] = {}
        token_ids: List[int] = []

        # Initial tokenization
        for token in pre_bpe_tokens:
            byte_encoded_token = tuple(token.encode("utf-8"))
            if byte_encoded_token not in dict_token_to_id:
                token_id = len(dict_token_to_id)
                dict_token_to_id[byte_encoded_token] = token_id
                dict_id_to_token[token_id] = byte_encoded_token
            else:
                token_id = dict_token_to_id[byte_encoded_token]
            token_ids.append(token_id)

        # Count token frequencies after pre-tokenization
        token_frequencies = Counter(token_ids)

        return dict_token_to_id, dict_id_to_token, token_ids, token_frequencies

    def _build_linked_lists_and_pair_info(self,
                                          token_frequencies: Dict[int, int],
                                          dict_id_to_token: Dict[int, Tuple[int, ...]]
                                          ) -> Tuple[
        List[ListNode], DefaultDict[Tuple[int, int], DefaultDict[int, List[Tuple[ListNode, ListNode]]]],
        DefaultDict[
            Tuple[int, int], int]]:
        """
        Builds linked lists and indexes the pair information and frequencies.

        Args:
            token_frequencies (Dict[int, int]): A map of token IDs to their frequencies.
            dict_id_to_token (Dict[int, List[int]]): A map of token IDs to their corresponding token sequences.

        Returns:
            Tuple[List[ListNode], DefaultDict[Tuple[int, int], DefaultDict[int, List[Tuple[ListNode, ListNode]]]], DefaultDict[Tuple[int, int], int]]:
                - A list of linked lists representing the token sequences.
                - A map that keeps track of node pairs and their locations.
                - A map of pair frequencies.
        """

        linked_lists: List[ListNode] = []
        pair_info_map: DefaultDict[
            Tuple[int, int], DefaultDict[int, List[Tuple[ListNode, ListNode]]]] = defaultdict(
            lambda: defaultdict(list))
        pair_frequencies: DefaultDict[Tuple[int, int], int] = defaultdict(int)

        for token_id, token_freq in token_frequencies.items():
            token = dict_id_to_token[token_id]

            head = ListNode(token[0])
            current = head
            linked_lists.append(head)

            for byte in token[1:]:
                new_node = ListNode(byte)
                current.next = new_node
                new_node.prev = current

                # Build pair_info_map and update pair_frequencies
                pair = (current.token, new_node.token)
                pair_info_map[pair][token_id].append((current, new_node))
                pair_frequencies[pair] += token_freq

                # Move to the next node
                current = new_node

        return linked_lists, pair_info_map, pair_frequencies

    def _merge_pair(self, next_index: int,
                    max_pair: Tuple[int, int],
                    pair_info_map: DefaultDict[Tuple[int, int], DefaultDict[int, List[Tuple[ListNode, ListNode]]]],
                    pair_frequencies: DefaultDict[Tuple[int, int], int],
                    token_frequencies: Dict[int, int],
                    linked_lists: List[ListNode]) -> None:

        # Get the list of node pairs (current, next) for each sequence from pair_info_map
        for token_id, node_pairs in pair_info_map[max_pair].items():
            for left_node, right_node in node_pairs:
                # Create the new merged node
                new_node = ListNode(next_index)
                new_node.prev, new_node.next = left_node.prev, right_node.next

                if left_node.prev:
                    left_node.prev.next = new_node
                else:
                    # If left_node was the head, update the head for the specific sequence
                    linked_lists[token_id] = new_node

                if right_node.next:
                    right_node.next.prev = new_node

                # Update pair_info_map and pair_frequencies for adjacent pairs & newly formed pairs
                self._update_pair_references_and_frequencies(pair_info_map,
                                                             pair_frequencies,
                                                             token_frequencies,
                                                             token_id,
                                                             left_node,
                                                             right_node,
                                                             new_node)

        # print(f"max_pair: {max_pair}, freq: {pair_frequencies[max_pair]}")
        # Remove the merged pair from pair_info_map and decrease its frequency to zero
        del pair_info_map[max_pair]
        del pair_frequencies[max_pair]

    def _update_pair_references_and_frequencies(self,
                                                pair_info_map: DefaultDict[
                                                    Tuple[int, int], DefaultDict[int, List[Tuple[ListNode, ListNode]]]],
                                                pair_frequencies: DefaultDict[Tuple[int, int], int],
                                                token_frequencies: Dict[int, int],
                                                token_id: int,
                                                left_node: ListNode,
                                                right_node: ListNode,
                                                new_node: ListNode) -> None:
        token_freq = token_frequencies[token_id]
        self._update_left_adjacent_pair(pair_info_map, pair_frequencies, token_freq, token_id, new_node, left_node)
        self._update_right_adjacent_pair(pair_info_map, pair_frequencies, token_freq, token_id, new_node, right_node)

    def _update_left_adjacent_pair(self, pair_info_map, pair_frequencies, token_freq, token_id, new_node, left_node):
        if new_node.prev:
            left_adjacent_pair_before_merge = (new_node.prev.token, left_node.token)
            left_adjacent_pair_after_merge = (new_node.prev.token, new_node.token)

            self._decrease_pair_frequency(pair_frequencies, left_adjacent_pair_before_merge, token_freq)
            self._remove_pair_from_map(pair_info_map, left_adjacent_pair_before_merge, token_id, new_node.prev,
                                       left_node)
            self._add_new_pair(pair_info_map, pair_frequencies, left_adjacent_pair_after_merge, token_id, token_freq,
                               new_node.prev, new_node)

    def _update_right_adjacent_pair(self, pair_info_map, pair_frequencies, token_freq, token_id, new_node, right_node):
        if new_node.next:
            right_adjacent_pair_before_merge = (right_node.token, new_node.next.token)
            right_adjacent_pair_after_merge = (new_node.token, new_node.next.token)

            self._decrease_pair_frequency(pair_frequencies, right_adjacent_pair_before_merge, token_freq)
            self._remove_pair_from_map(pair_info_map, right_adjacent_pair_before_merge, token_id, right_node,
                                       new_node.next)
            self._add_new_pair(pair_info_map, pair_frequencies, right_adjacent_pair_after_merge, token_id, token_freq,
                               new_node, new_node.next)

    def _decrease_pair_frequency(self, pair_frequencies, pair, token_freq):
        if pair in pair_frequencies:
            pair_frequencies[pair] -= token_freq
            if pair_frequencies[pair] == 0:
                del pair_frequencies[pair]

    def _remove_pair_from_map(self, pair_info_map, pair, token_id, left_node, right_node):
        if pair in pair_info_map and token_id in pair_info_map[pair]:
            pair_info_map[pair][token_id].remove((left_node, right_node))
            if not pair_info_map[pair][token_id]:
                del pair_info_map[pair][token_id]
                if not pair_info_map[pair]:
                    del pair_info_map[pair]

    def _add_new_pair(self, pair_info_map, pair_frequencies, new_pair, token_id, token_freq, left_node, right_node):
        pair_info_map[new_pair][token_id].append((left_node, right_node))
        pair_frequencies[new_pair] += token_freq

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
    trainer = BPETrainer(text, vocab_size, special_tokens)
    vocab, merges = trainer.train()
    return vocab, merges


if __name__ == '__main__':
    input_path = '../tests/FIXTURES/corpus.en'
    vocab, merges = run_train_bpe(input_path, 500, ['<PAD>'])
    print(vocab)

