import pytest
from chunking import split_text_with_overlap
from ingestion import extract_from_txt

def test_split_text_with_overlap():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = split_text_with_overlap(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 2
    assert all(len(c) <= 35 for c in chunks)

def test_txt_parser():
    mock_content = b"University Regulations for 2026."
    result = extract_from_txt(mock_content, "rules.txt")
    assert len(result) == 1
    assert result[0]["page"] == 1
    assert "University Regulations" in result[0]["text"]