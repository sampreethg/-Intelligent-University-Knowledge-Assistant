import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import database

INDEX_FILE = "faiss_index.bin"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model '{EMBEDDING_MODEL_NAME}': {str(e)}")
            
        self.dimension = 384
        
        if os.path.exists(INDEX_FILE):
            try:
                self.index = faiss.read_index(INDEX_FILE)
            except Exception:
                base_index = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIDMap2(base_index)
                self.save()
        else:
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap2(base_index)

    def save(self):
        try:
            faiss.write_index(self.index, INDEX_FILE)
        except Exception as e:
            print(f"[Error] Failed to persist FAISS index: {str(e)}")

    def add_documents(self, chunks: List[Dict[str, Any]], filename: str):
        if not chunks:
            return

        doc_id = database.record_document(filename)
        texts = [c["text"] for c in chunks if c.get("text", "").strip()]
        
        if not texts:
            return

        try:
            embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            current_total = self.index.ntotal
            assigned_ids = np.arange(current_total, current_total + len(texts), dtype=np.int64)

            self.index.add_with_ids(embeddings, assigned_ids)
            self.save()

            sqlite_payload = []
            for i, chunk in enumerate(chunks):
                sqlite_payload.append({
                    "id": int(assigned_ids[i]),
                    "doc_id": doc_id,
                    "filename": chunk.get("filename", filename),
                    "page_number": chunk.get("page_number", chunk.get("page", 1)),
                    "chunk_index": chunk.get("chunk_index", i),
                    "text_content": chunk.get("text", "")
                })
            database.insert_chunks(sqlite_payload)
        except Exception as e:
            self.delete_document(filename)
            raise RuntimeError(f"Indexing failed for '{filename}': {str(e)}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query.strip() or self.index.ntotal == 0:
            return []

        try:
            query_vector = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            valid_ids = []
            score_map = {}
            for idx, score in zip(indices[0], distances[0]):
                if idx != -1:
                    id_int = int(idx)
                    valid_ids.append(id_int)
                    # Convert cosine similarity (-1 to 1) to percentage clamp [0%, 100%]
                    similarity = float(score)
                    percentage = max(0.0, min(100.0, similarity * 100.0))
                    score_map[id_int] = round(percentage, 1)

            if not valid_ids:
                return []

            retrieved_records = database.get_chunks_by_ids(valid_ids)
            for record in retrieved_records:
                record["similarity_score"] = score_map.get(record["id"], 0.0)

            # Preserve descending rank order
            id_order = {valid_id: rank for rank, valid_id in enumerate(valid_ids)}
            retrieved_records.sort(key=lambda item: id_order.get(item["id"], 999))
            return retrieved_records
        except Exception as e:
            print(f"[Error] Search execution failed: {str(e)}")
            return []

    def delete_document(self, filename: str) -> int:
        try:
            chunk_ids = database.delete_document_by_name(filename)
            if chunk_ids and self.index.ntotal > 0:
                ids_to_remove = np.array(chunk_ids, dtype=np.int64)
                self.index.remove_ids(ids_to_remove)
                self.save()
            return len(chunk_ids)
        except Exception as e:
            print(f"[Error] Deletion failed for '{filename}': {str(e)}")
            return 0