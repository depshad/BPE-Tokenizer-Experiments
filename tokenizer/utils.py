import regex as re
from typing import List


def pre_tokenize(pattern: str, text: str) -> List[str]:
    tokens = re.findall(pattern, text)
    return tokens
