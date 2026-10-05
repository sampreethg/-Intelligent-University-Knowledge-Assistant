from typing import List, Dict

def split_text_with_overlap(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    if not text or not text.strip():
        return []
        
    text = text.strip()
    text_length = len(text)
    
    if text_length <= chunk_size:
        return [text]
        
    # Ensure overlap is non-negative and strictly less than chunk_size
    overlap = max(0, min(overlap, chunk_size - 1))
    
    chunks = []
    start = 0
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        # If not at the end of the text, break at the nearest newline or space
        if end < text_length:
            break_point = max(chunk.rfind("\n"), chunk.rfind(" "))
            if break_point > chunk_size // 2:
                end = start + break_point
                chunk = text[start:end]
        
        cleaned = chunk.strip()
        if cleaned:
            chunks.append(cleaned)
            
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
            
        start = next_start
        if start >= text_length:
            break
            
    return chunks

def process_extracted_pages(pages: List[Dict], chunk_size: int = 1500, overlap: int = 200) -> List[Dict]:
    all_chunks = []
    chunk_counter = 0
    for page in pages:
        page_num = page.get("page_number", page.get("page", 1))
        page_text = page.get("text", "")
        if not page_text or not page_text.strip():
            continue
        text_chunks = split_text_with_overlap(page_text, chunk_size, overlap)
        for chunk in text_chunks:
            all_chunks.append({
                "filename": page.get("filename", ""),
                "page_number": page_num,
                "page": page_num,
                "chunk_index": chunk_counter,
                "text": chunk
            })
            chunk_counter += 1
    return all_chunks