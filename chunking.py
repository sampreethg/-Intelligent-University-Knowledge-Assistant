from typing import List, Dict

def split_text_with_overlap(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # If not at the end of the text, break at the nearest newline or space
        if end < text_length:
            break_point = max(chunk.rfind("\n"), chunk.rfind(" "))
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point]
                end = start + break_point
        
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap
        if start >= text_length or end >= text_length:
            break
            
    return chunks

def process_extracted_pages(pages: List[Dict], chunk_size: int = 1500, overlap: int = 200) -> List[Dict]:
    all_chunks = []
    chunk_counter = 0
    for page in pages:
        page_num = page.get("page_number", page.get("page", 1))
        text_chunks = split_text_with_overlap(page.get("text", ""), chunk_size, overlap)
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