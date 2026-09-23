from typing import Optional, Tuple, Any

from .translator_google import GoogleTranslator
from .translator_transformers import TransformersTranslator
from .translator_llama import LlamaCppTranslator


def create_translator(translator_type: str, target_lang: str, llama_url: str = "http://localhost:8080/v1",
                      llama_model: Optional[str] = None, auto_find: bool = True,
                      transformers_model: Optional[str] = None) -> Tuple[Any, Any]:
    if translator_type == "llama":
        return LlamaCppTranslator(target_lang, api_base=llama_url, expected_model=llama_model, auto_find=auto_find), None
    elif translator_type == "transformers":
        return TransformersTranslator(target_lang, model=transformers_model), None
    elif translator_type == "google":
        primary = GoogleTranslator(target_lang)
        fallback = None
        try:
            fallback = LlamaCppTranslator(target_lang, api_base=llama_url, expected_model=llama_model, auto_find=auto_find)
        except Exception:
            pass
        if not fallback:
            try:
                fallback = TransformersTranslator(target_lang, model=transformers_model)
            except Exception:
                pass
        return primary, fallback
    else:
        raise ValueError(f"Неизвестный переводчик: {translator_type}")
