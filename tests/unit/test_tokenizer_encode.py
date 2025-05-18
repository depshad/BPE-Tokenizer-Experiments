import pytest
from tokenizer.tokenizer import Tokenizer

def test_single_character(tokenizer):
    """Test tokenization of single character"""
    result = tokenizer.encode('a')
    assert result == [1]

def test_no_merges_possible(tokenizer):
    """Test when no merges are possible"""
    result = tokenizer.encode('tbe')
    assert result == [10, 2, 5]

def test_single_merge(tokenizer):
    """Test single merge operation"""
    result = tokenizer.encode('ea')
    assert result == [11]   # 'ea' merged

def test_multiple_merges(tokenizer):
    """Test multiple sequential merges"""
    result = tokenizer.encode('eating')
    assert result == [12, 13, 7]  # ['eat', 'in', 'g']

def test_priority_order(tokenizer):
    """Test merges happen in correct priority order"""
    result = tokenizer.encode('abd')
    assert result == [15, 4]  # Should merge a,b first due to priority

def test_empty_string(tokenizer):
    """Test empty input"""
    result = tokenizer.encode('')
    assert result == []

def test_unknown_character(tokenizer):
    """Test handling of unknown characters"""
    with pytest.raises(KeyError):
        tokenizer.encode('x')

def test_overlapping_merges(tokenizer):
    """Test correct handling of overlapping merge possibilities"""
    result = tokenizer.encode('abc')
    assert result == [18]  # Should result in 'abc'

def test_merge_at_start(tokenizer):
    """Test merging at the start of sequence"""
    result = tokenizer.encode('eat')
    assert result == [12]  # 'eat'

def test_merge_at_end(tokenizer):
    """Test merging at the end of sequence"""
    result = tokenizer.encode('tin')
    assert result == [10, 13]  # ['t', 'in']

def test_repeated_patterns(tokenizer):
    """Test handling of repeated patterns"""
    result = tokenizer.encode('aaa')
    # Expected: first two 'a's merge to 'aa', then the last 'a' remains
    assert result == [17, 1]