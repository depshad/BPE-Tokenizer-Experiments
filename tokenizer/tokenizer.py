import logging
from typing import List
import regex as re
from heapq import heappush, heappop
from itertools import count


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
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.inverse_vocab = {v: k for k, v in self.vocab.items()} # map from token -> token_id
        self.merges_id = self.create_merges_id()

    def pre_tokenize(self,pattern: str, text:str) -> List[str]:
        tokens = re.findall(pattern, text)
        return tokens

    def create_merges_id(self):
        """
        Creates a dictionary mapping merged token IDs to their new token ID.

        Args:
            vocab (dict): A dictionary mapping token IDs to their token strings.
            merges (list): A list of tuples representing byte pair merges.

        Returns:
            dict: A dictionary where keys are (token_id_1, token_id_2) and values are the merged token_id.
        """
        inverse_vocab = {v: k for k, v in self.vocab.items()}  # Reverse vocab lookup
        merges_id = {}

        for merge in self.merges:
            token_1, token_2 = merge  # Byte pair merge

            # Convert bytes to strings if necessary
            if isinstance(token_1, bytes):
                token_1 = token_1.decode("utf-8")
            if isinstance(token_2, bytes):
                token_2 = token_2.decode("utf-8")

            # Ensure both tokens exist in inverse_vocab
            if token_1 in inverse_vocab and token_2 in inverse_vocab:
                id_1 = inverse_vocab[token_1]
                id_2 = inverse_vocab[token_2]
                merged_token = token_1 + token_2

                # Assign the next available token ID based on vocab size
                if merged_token in inverse_vocab:
                    merged_id = inverse_vocab[merged_token]  # Use existing ID if available
                else:
                    merged_id = max(self.vocab.keys()) + 1  # Assign new token ID

                # Store the merge result
                merges_id[(id_1, id_2)] = merged_id

                # Add to vocab (update dynamically)
                self.vocab[merged_id] = merged_token
                inverse_vocab[merged_token] = merged_id  # Keep vocab in sync

        return merges_id

    def encode(self, text):
        """
        Encode the input text into a list of token IDs.
        Args:
            text (str): The text to encode.
        Returns:
            List[int]: The list of token IDs.
        """
        tokens = []
        # Pre-tokenize the text
        words = self.pre_tokenize(pattern=self.GPT2_PAT, text=text)
        for i, word in enumerate(words):
            if word.isspace():
                tokens.append(word)  # Preserve exact whitespace tokens
            elif i > 0 and words[i - 1].isspace():
                tokens.append("Ġ" + word)  # Prefix with "Ġ" if previous token was space
            else:
                tokens.append(word)  # No need to add "Ġ" if first word
        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                # token is contained in the vocabulary as is
                token_id = self.inverse_vocab[token]
                token_ids.append(token_id)
            else:
                # Attempt to handle subword tokenization via BPE
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)
        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # ToDo Merge the pairs based on the priority order (i.e. most frequent pair first), this can be derived from the vocab order (lower the order the better)
        # Or we can also output pair frequency in merges
        class TokenNode:
            def __init__(self, token_id=None):
                self.token_id = token_id
                self.next = None
                self.prev = None
        # Create dummy head
        dummy_head = TokenNode()
        curr = dummy_head
        # Build linked list starting after dummy head
        for char in token:
            token_id = self.inverse_vocab[char]
            new_node = TokenNode(token_id)
            curr.next = new_node
            new_node.prev = curr
            curr = new_node
        # Initial merge candidates
        merge_candidates = []
        curr = dummy_head.next  # Start from first real node
        merge_order = count()  # This ensures a unique, always-increasing counter

        while curr and curr.next:
            pair = (curr.token_id, curr.next.token_id)
            if pair in self.merges_id:
                merged_id = self.merges_id[pair]
                heappush(merge_candidates, (merged_id, next(merge_order), curr)) # Lower the merged_id, better the priority
            curr = curr.next
        while merge_candidates:
            merged_id, _, left_node = heappop(merge_candidates)
            # Validate merge is still possible
            if not left_node.next:
                continue
            pair = (left_node.token_id, left_node.next.token_id)
            if pair not in self.merges_id:
                continue
            # Apply merge
            right_node = left_node.next
            left_node.token_id = merged_id
            # Update links
            left_node.next = right_node.next
            if right_node.next:
                right_node.next.prev = left_node
            # Check new possible merges
            # Safe to check prev because of dummy head
            if left_node.prev != dummy_head:  # Don't merge with dummy head
                new_pair = (left_node.prev.token_id, left_node.token_id)
                if new_pair in self.merges_id:
                    new_merged_id = self.merges_id[new_pair]
                    heappush(merge_candidates, (new_merged_id, next(merge_order), left_node.prev))
            if left_node.next:
                new_pair = (left_node.token_id, left_node.next.token_id)
                if new_pair in self.merges_id:
                    new_merged_id = self.merges_id[new_pair]
                    heappush(merge_candidates, (new_merged_id, next(merge_order), left_node))
        # Convert to result list, skipping dummy head
        result = []
        curr = dummy_head.next
        while curr:
            result.append(curr.token_id)
            curr = curr.next
        return result

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.
        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token.startswith("Ġ"):
                # Replace 'Ġ' with a space
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

def get_tokenizer(vocab, merges, special_tokens):
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer