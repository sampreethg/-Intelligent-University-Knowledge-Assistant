import os
import pytest
from chunking import split_text_with_overlap, process_extracted_pages
from ingestion import parse_document, extract_from_txt
import database
from vector_store import VectorStore
from llm import stream_grounded_answer

def test_split_text_with_overlap():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = split_text_with_overlap(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 2
    assert all(len(c) <= 35 for c in chunks)

def test_split_text_edge_cases():
    # Empty string
    assert split_text_with_overlap("") == []
    assert split_text_with_overlap("   ") == []

    # Short string less than chunk size
    short_text = "Short syllabus clause."
    assert split_text_with_overlap(short_text, chunk_size=100) == [short_text]

    # Large overlap that previously caused infinite loops
    stress_text = "1234567890 1234567890 1234567890 1234567890"
    chunks = split_text_with_overlap(stress_text, chunk_size=30, overlap=28)
    assert len(chunks) > 0

    # Overlap greater than or equal to chunk_size
    chunks_capped = split_text_with_overlap(stress_text, chunk_size=20, overlap=30)
    assert len(chunks_capped) > 0

def test_process_extracted_pages():
    pages = [
        {"filename": "doc.pdf", "page_number": 1, "text": "Page one text content."},
        {"filename": "doc.pdf", "page": 2, "text": "Page two text content."},
        {"filename": "doc.pdf", "page_number": 3, "text": "   "}  # empty page
    ]
    chunks = process_extracted_pages(pages, chunk_size=100, overlap=20)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["chunk_index"] == 1

def test_txt_parser():
    mock_content = b"University Regulations for 2026."
    result = extract_from_txt(mock_content, "rules.txt")
    assert len(result) == 1
    assert result[0]["page_number"] == 1
    assert result[0]["page"] == 1
    assert "University Regulations" in result[0]["text"]

def test_ingestion_latin1_fallback():
    latin1_bytes = "Café regulations and naïve policy".encode("latin-1")
    result = parse_document(latin1_bytes, "policy.txt")
    assert len(result) == 1
    assert "Café regulations" in result[0]["text"]

def test_ingestion_empty_and_invalid():
    with pytest.raises(ValueError, match="is empty"):
        parse_document(b"", "empty.txt")

    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_document(b"random binary data", "archive.zip")

def test_database_crud(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_knowledge.db")
    monkeypatch.setattr(database, "DB_PATH", test_db)

    # Initialize DB schema
    database.init_db()

    # Next chunk ID should start at 0
    assert database.get_next_chunk_id() == 0
    assert not database.document_exists("academic_policy.pdf")

    # Record document
    doc_id = database.record_document("academic_policy.pdf")
    assert doc_id > 0
    assert database.document_exists("academic_policy.pdf")

    # Insert chunks
    chunks_payload = [
        {
            "id": 0,
            "doc_id": doc_id,
            "filename": "academic_policy.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text_content": "Minimum attendance required is 75 percent."
        },
        {
            "id": 1,
            "doc_id": doc_id,
            "filename": "academic_policy.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "text_content": "Condonation is granted up to 65 percent with medical certificate."
        }
    ]
    database.insert_chunks(chunks_payload)
    assert database.get_next_chunk_id() == 2

    # Query chunks by ID
    retrieved = database.get_chunks_by_ids([0, 1])
    assert len(retrieved) == 2
    assert retrieved[0]["chunk_index"] == 0
    assert "attendance" in retrieved[0]["text_content"]

    # Preview chunks by filename
    preview = database.get_chunks_by_filename("academic_policy.pdf")
    assert len(preview) == 2

    # List documents
    docs = database.list_uploaded_documents()
    assert "academic_policy.pdf" in docs

    # Delete document
    deleted_ids = database.delete_document_by_name("academic_policy.pdf")
    assert set(deleted_ids) == {0, 1}
    assert not database.document_exists("academic_policy.pdf")
    assert database.list_uploaded_documents() == []

def test_vector_store_workflow_and_collision_prevention(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_vs.db")
    test_index = str(tmp_path / "test_faiss.bin")

    monkeypatch.setattr(database, "DB_PATH", test_db)
    monkeypatch.setattr("vector_store.INDEX_FILE", test_index)
    database.init_db()

    store = VectorStore()
    assert store.index.ntotal == 0

    # 1. Add Doc A (2 chunks)
    doc_a_chunks = [
        {"filename": "doc_a.txt", "page_number": 1, "chunk_index": 0, "text": "Computer Science syllabus covers data structures and algorithms."},
        {"filename": "doc_a.txt", "page_number": 1, "chunk_index": 1, "text": "Operating systems and database management systems are core subjects."}
    ]
    store.add_documents(doc_a_chunks, "doc_a.txt")
    assert store.index.ntotal == 2

    # 2. Add Doc B (2 chunks)
    doc_b_chunks = [
        {"filename": "doc_b.txt", "page_number": 1, "chunk_index": 0, "text": "Mechanical engineering covers thermodynamics and fluid mechanics."},
        {"filename": "doc_b.txt", "page_number": 1, "chunk_index": 1, "text": "Machine design and manufacturing technology require laboratory practicals."}
    ]
    store.add_documents(doc_b_chunks, "doc_b.txt")
    assert store.index.ntotal == 4

    # 3. Delete Doc A -> index total becomes 2, max remaining ID in DB is 3
    removed_count = store.delete_document("doc_a.txt")
    assert removed_count == 2
    assert store.index.ntotal == 2

    # 4. Add Doc C (2 chunks) -> Must not collide with Doc B's existing IDs!
    doc_c_chunks = [
        {"filename": "doc_c.txt", "page_number": 1, "chunk_index": 0, "text": "Electrical engineering covers circuit analysis and power systems."},
        {"filename": "doc_c.txt", "page_number": 1, "chunk_index": 1, "text": "Control systems and electrical machines require minimum passing marks of 50."}
    ]
    store.add_documents(doc_c_chunks, "doc_c.txt")
    assert store.index.ntotal == 4

    # 5. Search validation
    results = store.search("thermodynamics fluid mechanics", top_k=2)
    assert len(results) > 0
    assert results[0]["filename"] == "doc_b.txt"
    assert 0.0 <= results[0]["similarity_score"] <= 100.0

    # 6. Re-uploading doc_b replaces it cleanly
    store.add_documents(doc_b_chunks, "doc_b.txt")
    assert store.index.ntotal == 4

def test_llm_edge_cases():
    # Empty query generator
    gen1 = stream_grounded_answer("", [{"filename": "f.txt", "page_number": 1, "text_content": "text"}])
    assert "Query cannot be empty" in "".join(list(gen1))

    # Empty context generator
    gen2 = stream_grounded_answer("What is the fee?", [])
    assert "does not contain any relevant documents" in "".join(list(gen2))