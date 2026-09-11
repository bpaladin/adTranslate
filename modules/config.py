"""Configuration loading for PDF Translator."""

import os
import json
from pathlib import Path

from .logging_setup import logger


def _parse_env_file(path: Path):
    try:
        if not path.exists():
            return
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception as e:
        logger.warning(f"Ошибка загрузки {path.name}: {e}")


def load_env_file():
    script_dir = Path(__file__).parent.parent.absolute()
    for name in ('keys.env', '.env'):
        _parse_env_file(script_dir / name)


load_env_file()

DEFAULT_SETTINGS = {
    "block_timeout": 900,
    "request_timeout": 120,
    "api_timeout": 120,
    "model_cooldown_sec": 60,
    "llama_max_retries": 3,
    "llm_chunk_chars": 1800,
    "llm_min_chunk_chars": 650,
    "llm_rescue_min_chars": 220,
    "max_section_chars": 4000,
    "max_llm_chunk_chars": 2500,
    "min_llm_chunk_chars": 650,
}


def load_settings(path: str = "settings.json") -> dict:
    _config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    p = Path(path)
    if not p.is_absolute():
        p = Path(os.path.join(_config_dir, path))
    if p.exists():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            merged = dict(DEFAULT_SETTINGS)
            merged.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
            return merged
        except Exception as e:
            logger.warning(f"Ошибка чтения {p.name}: {e} — используются умолчания")
    try:
        os.makedirs(_config_dir, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        logger.info(f"Создан файл настроек: {p.name}")
    except Exception as e:
        logger.debug(f"Не удалось создать {p.name}: {e}")
    return dict(DEFAULT_SETTINGS)


SETTINGS = load_settings()
BLOCK_TIMEOUT = int(SETTINGS["block_timeout"])
REQUEST_TIMEOUT = max(120, int(SETTINGS["request_timeout"]))
API_TIMEOUT = SETTINGS["api_timeout"]
MODEL_COOLDOWN_SEC = SETTINGS["model_cooldown_sec"]
LLAMA_MAX_RETRIES = SETTINGS["llama_max_retries"]
MAX_SECTION_CHARS = int(SETTINGS["max_section_chars"])
MAX_LLM_CHUNK_CHARS = int(SETTINGS["max_llm_chunk_chars"])
MIN_LLM_CHUNK_CHARS = int(SETTINGS["min_llm_chunk_chars"])

OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://sprutdock.ru/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "z-ai/glm-5.2:free")
SPRUTDOCK_API_KEY = os.getenv("SPRUTDOCK_API_KEY") or OPENAI_API_KEY
