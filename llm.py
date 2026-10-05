import os
from openai import OpenAI, APIConnectionError, APIError
from typing import List, Dict, Any, Generator

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    timeout=60.0  # Avoid hanging indefinitely on CPU inference
)

def stream_grounded_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
    if not query.strip():
        yield "Query cannot be empty."
        return

    if not context_chunks:
        yield "The institutional knowledge base does not contain any relevant documents to answer this question."
        return

    formatted_context = []
    for chunk in context_chunks:
        formatted_context.append(
            f"[Source: {chunk.get('filename', 'Unknown')}, Page {chunk.get('page_number', '?')}]:\n{chunk.get('text_content', '')}"
        )
    context_str = "\n\n---\n\n".join(formatted_context)

    system_prompt = (
        "You are an Intelligent University Knowledge Assistant. "
        "Answer the user's question using ONLY the provided context excerpts below. "
        "Rules:\n"
        "1. Every factual statement must cite its source in the format [Filename, Page X].\n"
        "2. If the answer cannot be determined strictly from the context, state: "
        "'The provided institutional documents do not contain information to answer this question.'\n"
        "3. Do not assume or extrapolate facts not explicitly present in the text."
    )

    user_message = f"Retrieved Context:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except APIConnectionError:
        yield (
            "⚠️ **Connection Error**: Unable to reach the local Ollama engine at `http://localhost:11434`. "
            "Please ensure Ollama is active by executing `ollama serve` in your terminal."
        )
    except APIError as api_err:
        if "model" in str(api_err).lower():
            yield f"⚠️ **Model Error**: The specified model `{MODEL_NAME}` was not found. Run `ollama pull {MODEL_NAME}`."
        else:
            yield f"⚠️ **Inference Error**: {str(api_err)}"
    except Exception as e:
        yield f"⚠️ **Unexpected Error during inference**: {str(e)}"