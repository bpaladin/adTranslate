import time
import logging
from typing import Optional, Dict, Any, List

import markdown as md

from .translator_factory import create_translator
from .utils import _chunk_text_by_sentences
from .translation_cache import TranslationCache

logger = logging.getLogger(__name__)


def generate_summary(pages: list, summary_translator_type: str, translate_translator_type: str,
                     lang: str, llama_url: str = "http://localhost:8080/v1",
                     llama_model: Optional[str] = None, auto_find: bool = True,
                     sprut_model: Optional[str] = None, quiet: bool = False) -> Dict[str, Any]:
    all_text = []
    for page in pages:
        for block in page.blocks:
            if block.type in ("paragraph", "heading", "metadata", "text", "list"):
                text = block.text
                if text and len(text) > 10:
                    all_text.append(text)
    full_text = "\n".join(all_text)
    if not full_text:
        return {"summary_html": "<p>Не удалось извлечь текст для реферата.</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0}}

    if summary_translator_type == "google":
        summary_translator_type = "llama"

    try:
        summary_translator, _ = create_translator(summary_translator_type, "en", llama_url=llama_url,
                                                   llama_model=llama_model, auto_find=auto_find, sprut_model=sprut_model)
    except Exception as e:
        return {"summary_html": f"<p>Ошибка LLM: {e}</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": str(e)}}

    try:
        final_translator, _ = create_translator(translate_translator_type, lang, llama_url=llama_url,
                                                 llama_model=llama_model, auto_find=auto_find, sprut_model=sprut_model)
    except Exception as e:
        return {"summary_html": f"<p>Ошибка переводчика: {e}</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": str(e)}}

    chunks = _chunk_text_by_sentences(full_text, max_chunk_size=6000)
    cache = TranslationCache()
    cache_key = f"summary:{summary_translator_type}"
    chunk_summaries = []
    start_time = time.time()
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        cached = cache.get(chunk, cache_key)
        if cached:
            chunk_summaries.append(cached)
            continue
        prompt = "Extract key facts from this text for a summary. Write the summary in English:\n" + chunk
        summary = summary_translator.generate(prompt)
        if summary:
            cache.put(chunk, cache_key, summary)
            chunk_summaries.append(summary)
            total_tokens += len(summary) // 4

    if not chunk_summaries:
        return {"summary_html": "<p>Не удалось сгенерировать реферат.</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0}}

    summaries_to_combine = chunk_summaries
    if len(chunk_summaries) > 5:
        grouped = []
        for j in range(0, len(chunk_summaries), 5):
            group = chunk_summaries[j:j + 5]
            group_prompt = "Combine these key points into a concise summary:\n" + "\n---\n".join(group)
            group_summary = summary_translator.generate(group_prompt)
            grouped.append(group_summary if group_summary else "\n".join(group))
        summaries_to_combine = grouped

    final_prompt = ("Based on the following key points, create a final structured summary in English strictly in the format:\n"
                    "BRIEF CONTENT: 4-5 sentences\nKEY FINDINGS: list\nSTRENGTHS: list\nWEAKNESSES: list\n\nKey points:\n"
                    + "\n---\n".join(summaries_to_combine))
    summary_en = summary_translator.generate(final_prompt) or "\n\n".join(summaries_to_combine)

    try:
        final_translated = final_translator.translate(summary_en)
        summary_html = md.markdown(final_translated if final_translated else summary_en)
    except Exception:
        summary_html = md.markdown(summary_en)

    elapsed = time.time() - start_time
    speed = total_tokens / elapsed if elapsed > 0 else 0
    return {"summary_html": summary_html,
            "stats": {"chunks": len(chunks), "tokens": total_tokens, "speed": speed,
                      "model": getattr(summary_translator, '_loaded_model', summary_translator.name), "time": elapsed}}
