import os
import sqlite3
from typing import List, Dict, Any, Tuple

DB_PATH = os.getenv("DATABASE_PATH", "documents.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER,
                filename TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                text_content TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """)
        conn.commit()

def record_document(filename: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO documents (filename) VALUES (?)", (filename,))
        conn.commit()
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        return cursor.fetchone()["id"]

def insert_chunks(chunks_data: List[Dict[str, Any]]):
    with get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO chunks (id, doc_id, filename, page_number, chunk_index, text_content)
            VALUES (:id, :doc_id, :filename, :page_number, :chunk_index, :text_content)
        """, chunks_data)
        conn.commit()

def get_chunks_by_ids(ids: List[int]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, filename, page_number, text_content 
            FROM chunks 
            WHERE id IN ({placeholders})
        """, ids)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def list_uploaded_documents() -> List[str]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM documents ORDER BY upload_timestamp DESC")
        return [row["filename"] for row in cursor.fetchall()]

def delete_document_by_name(filename: str) -> List[int]:
    with get_connection() as conn:
        cursor = conn.cursor()
        # 1. Fetch chunk IDs to remove from FAISS
        cursor.execute("SELECT id FROM chunks WHERE filename = ?", (filename,))
        chunk_ids = [row["id"] for row in cursor.fetchall()]

        # 2. Delete document (foreign key cascades or explicit delete)
        cursor.execute("DELETE FROM chunks WHERE filename = ?", (filename,))
        cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        conn.commit()

        return chunk_ids
def get_chunks_by_filename(filename: str) -> List[Dict[str, Any]]:
    """Retrieve all chunk records for a given document name for previewing."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, page_number, chunk_index, text_content 
            FROM chunks 
            WHERE filename = ? 
            ORDER BY page_number ASC, chunk_index ASC
            """,
            (filename,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]