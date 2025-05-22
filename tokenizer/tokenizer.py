import logging
from typing import List, Tuple, Dict, Optional
from heapq import heappush, heappop
from itertools import count
from tokenizer.utils import pre_tokenize

logger = logging.getLogger(__name__)

# Configure logging at the top level of your script
def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,  # Change to INFO or WARNING in production
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()  # Console handler
            # Add FileHandler or other handlers if needed
        ]
    )


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens, pair_to_merged_id=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.pair_to_merged_id = pair_to_merged_id
        if pair_to_merged_id is None:
            self.pair_to_merged_id = self._calculate_pair_to_merged_id(vocab, merges)
            logger.info("pair_to_merged_id was not supplied, calculated it from vocab and merges.")

    @staticmethod
    def _calculate_pair_to_merged_id(
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],
    ) -> Dict[Tuple[int, int], int]:
        """
        Reconstructs the pair_to_merged_id mapping from the final vocabulary and ordered merge rules.

        This function simulates the BPE merge process to determine which pair of token IDs
        resulted in each new merged token ID. The order in the `merges` list is crucial,
        as it defines the sequential assignment of new token IDs (starting from 256).
        Returns:
            A dictionary mapping (token_id1, token_id2) to their merged_token_id.
        """
        pair_to_merged_id: Dict[Tuple[int, int], int] = {}

        # Initialize a working inverse vocabulary (bytes -> ID).
        # Start with the base byte tokens (0-255).
        inverse_vocab: Dict[bytes, int] = {bytes([i]): i for i in range(256)}

        # Determine the starting ID for new merged tokens.
        # This is typically 256 for standard BPE.
        next_new_token_id = 256

        # Iterate through the merges list in the order they occurred.
        for idx, (byte1_sequence, byte2_sequence) in enumerate(merges):
            merged_token_id = next_new_token_id
            merged_bytes_representation = byte1_sequence + byte2_sequence

            # Sanity check: Ensure this `merged_token_id` maps to the expected
            # `merged_bytes_representation` in the final `vocab`.
            if merged_token_id not in vocab:
                logger.warning(f"Reconstruction Warning: Merge {idx}: Expected new token ID {merged_token_id} "
                               f"not found in final vocab. This suggests an inconsistent model state.")
                # Decide how to handle this: raise an error, or try to infer, etc.
                # For robustness, we'll continue but this indicates a potential issue.
            elif vocab[merged_token_id] != merged_bytes_representation:
                logger.warning(f"Reconstruction Warning: Merge {idx}: Vocab entry {vocab[merged_token_id]} "
                               f"for ID {merged_token_id} does not match expected merged bytes "
                               f"{merged_bytes_representation}. Model inconsistency suspected.")

            # Get the token IDs for the two components being merged from the current working vocab.
            id1 = inverse_vocab.get(byte1_sequence)
            id2 = inverse_vocab.get(byte2_sequence)

            if id1 is None:
                logger.error(f"Reconstruction Error: Component '{byte1_sequence}' (first part of merge {idx}) "
                             f"not found in current working vocabulary. Cannot reconstruct this merge.")
                continue
            if id2 is None:
                logger.error(f"Reconstruction Error: Component '{byte2_sequence}' (second part of merge {idx}) "
                             f"not found in current working vocabulary. Cannot reconstruct this merge.")
                continue

            # Add the derived pair-to-merged-ID mapping.
            pair_to_merged_id[(id1, id2)] = merged_token_id

            # Update `current_inverse_vocab`: The new merged token is now available for subsequent merges.
            inverse_vocab[merged_bytes_representation] = merged_token_id

            next_new_token_id += 1

        return pair_to_merged_id

    def encode(self, text):
        """
        Encode the input text into a list of token IDs.
        Args:
            text (str): The text to encode.
        Returns:
            List[int]: The list of token IDs.
        """
        all_token_ids: List[int] = []
        # Pre-tokenize the text
        pre_tokens = pre_tokenize(pattern=self.GPT2_PAT, text=text)

        for pre_token in pre_tokens:
            pre_token_bytes = pre_token.encode('utf-8')
            if pre_token_bytes in self.inverse_vocab:
                # If the pre-token is in the vocab, add its ID directly
                all_token_ids.append(self.inverse_vocab[pre_token_bytes])
                continue

            bpe_ids_for_pre_token = self._apply_bpe_to_pre_token(pre_token_bytes)
            all_token_ids.extend(bpe_ids_for_pre_token)

        return all_token_ids

    def _apply_bpe_to_pre_token(self, pre_token_bytes:bytes):
        """
        Tokenize a word into BPE IDs using a heap for efficiency.

        Args:
            pre_token_bytes (bytes): The word to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Convert the word to bytes
        logger.debug(f"Applying BPE to pre-token bytes: {pre_token_bytes}")
        # Initialize the list of token IDs. Each byte becomes its own token ID (0-255).
        # Example: b"hello" -> [104, 101, 108, 108, 111]
        current_token_ids: List[int] = list(pre_token_bytes)

        # Main BPE merge loop for this segment
        # Continue merging as long as there are at least two tokens and a merge is found.
        while len(current_token_ids) > 1:
            best_merge_idx: int = -1
            min_merged_id: int = float('inf')  # Use merged_id as priority (lower ID = higher priority)

            # Iterate through all possible pairs in the current sequence of token IDs
            for i in range(len(current_token_ids) - 1):
                pair: Tuple[int, int] = (current_token_ids[i], current_token_ids[i + 1])

                # Check if this pair is a known merge candidate
                if pair in self.pair_to_merged_id:
                    current_pair_merged_id = self.pair_to_merged_id[pair]

                    # If this merge has a higher priority (lower merged_id) than what we've found so far,
                    # or if it's the first candidate found, update our best merge.
                    if current_pair_merged_id < min_merged_id:
                        min_merged_id = current_pair_merged_id
                        best_merge_idx = i

            # If no mergeable pair was found in this iteration, we stop.
            if best_merge_idx == -1:
                break

            # Apply the best merge:
            # Replace the two tokens at `best_merge_idx` and `best_merge_idx + 1`
            # with their newly merged token ID.
            new_token_id = min_merged_id
            current_token_ids = (
                    current_token_ids[:best_merge_idx] +
                    [new_token_id] +
                    current_token_ids[best_merge_idx + 2:]
            )
            logger.debug(
                f"Merged pair at index {best_merge_idx} into ID {new_token_id}. New sequence: {current_token_ids}")

        return current_token_ids




    # def tokenize_with_bpe(self, token):
    #     """
    #     Tokenize a single token using BPE merges.
    #
    #     Args:
    #         token (str): The token to tokenize.
    #
    #     Returns:
    #         List[int]: The list of token IDs after applying BPE.
    #     """
    #     # ToDo Merge the pairs based on the priority order (i.e. most frequent pair first), this can be derived from the vocab order (lower the order the better)
    #     # Or we can also output pair frequency in merges
    #
    #     # Create dummy head
    #     dummy_head = TokenNode()
    #     curr = dummy_head
    #     # Build linked list starting after dummy head
    #     for char in token:
    #         # Convert char to byte
    #
    #         token_id = self.inverse_vocab[char]
    #         new_node = TokenNode(token_id)
    #         curr.next = new_node
    #         new_node.prev = curr
    #         curr = new_node
    #     # Initial merge candidates
    #     merge_candidates = []
    #     curr = dummy_head.next  # Start from first real node
    #     merge_order = count()  # This ensures a unique, always-increasing counter
    #
    #     while curr and curr.next:
    #         pair = (curr.token_id, curr.next.token_id)
    #         if pair in self.pair_to_merged_id:
    #             merged_id = self.pair_to_merged_id[pair]
    #             heappush(merge_candidates, (merged_id, next(merge_order), curr)) # Lower the merged_id, better the priority
    #         curr = curr.next
    #     while merge_candidates:
    #         merged_id, _, left_node = heappop(merge_candidates)
    #         # Validate merge is still possible
    #         if not left_node.next:
    #             continue
    #         pair = (left_node.token_id, left_node.next.token_id)
    #         if pair not in self.pair_to_merged_id:
    #             continue
    #         # Apply merge
    #         right_node = left_node.next
    #         left_node.token_id = merged_id
    #         # Update links
    #         left_node.next = right_node.next
    #         if right_node.next:
    #             right_node.next.prev = left_node
    #         # Check new possible merges
    #         # Safe to check prev because of dummy head
    #         if left_node.prev != dummy_head:  # Don't merge with dummy head
    #             new_pair = (left_node.prev.token_id, left_node.token_id)
    #             if new_pair in self.pair_to_merged_id:
    #                 new_merged_id = self.pair_to_merged_id[new_pair]
    #                 heappush(merge_candidates, (new_merged_id, next(merge_order), left_node.prev))
    #         if left_node.next:
    #             new_pair = (left_node.token_id, left_node.next.token_id)
    #             if new_pair in self.pair_to_merged_id:
    #                 new_merged_id = self.pair_to_merged_id[new_pair]
    #                 heappush(merge_candidates, (new_merged_id, next(merge_order), left_node))
    #     # Convert to result list, skipping dummy head
    #     result = []
    #     curr = dummy_head.next
    #     while curr:
    #         result.append(curr.token_id)
    #         curr = curr.next
    #     return result

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            token_ids: The list of token IDs to decode.

        Returns:
            The decoded string.
        """
        logger.debug(f"Starting decoding for IDs: {token_ids[:10]}...")
        decoded_bytes_parts: List[bytes] = []

        for token_id in token_ids:
            if token_id not in self.vocab:
                logger.warning(f"Token ID {token_id} not found in vocabulary. Skipping.")
                continue
            decoded_bytes_parts.append(self.vocab[token_id])

        # Concatenate all byte parts and decode to string
        full_byte_sequence = b"".join(decoded_bytes_parts)

        # Decode using UTF-8. 'errors=replace' handles any invalid byte sequences gracefully.
        decoded_string = full_byte_sequence.decode('utf-8', errors='replace')
        logger.debug("Decoding complete.")
        return decoded_string




def get_tokenizer(vocab, merges, special_tokens):
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer