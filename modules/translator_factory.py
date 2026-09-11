from typing import Optional, Tuple, Any

from .translator_google import GoogleTranslator
from .translator_sprut import SprutRotator
from .translator_llama import LlamaCppTranslator
from .config import OPENAI_API_KEY


def create_translator(translator_type: str, target_lang: str, llama_url: str = "http://localhost:8080/v1",
                      llama_model: Optional[str] = None, auto_find: bool = True,
                      sprut_model: Optional[str] = None) -> Tuple[Any, Any]:
    if translator_type == "llama":
        return LlamaCppTranslator(target_lang, api_base=llama_url, expected_model=llama_model, auto_find=auto_find), None
    elif translator_type in ("openrouter", "sprut"):
        return SprutRotator(target_lang, model=sprut_model), None
    elif translator_type == "google":
        primary = GoogleTranslator(target_lang)
        fallback = None
        if OPENAI_API_KEY:
            try:
                or_tr = SprutRotator(target_lang, model=sprut_model)
                if or_tr._client:
                    fallback = or_tr
            except Exception:
                pass
        if not fallback:
            try:
                fallback = LlamaCppTranslator(target_lang, api_base=llama_url, expected_model=llama_model, auto_find=auto_find)
            except Exception:
                pass
        return primary, fallback
    else:
        raise ValueError(f"Неизвестный переводчик: {translator_type}")
