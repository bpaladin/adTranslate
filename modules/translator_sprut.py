"""SprutDock translator backend for PDF Translator."""

import time
from typing import Optional, Dict

from .logging_setup import logger
from .rate_limiter import RateLimiter
from .translation_utils import _translation_system_prompt
from .utils import _interruptible_sleep, STOP_EVENT
from .config import DEFAULT_MODEL, SPRUTDOCK_API_KEY, OPENAI_BASE_URL, MODEL_COOLDOWN_SEC, API_TIMEOUT


class SprutRotator:
    def __init__(self, target_lang: str, model: Optional[str] = None):
        self.target_lang = target_lang
        self.name = "SprutDock"
        self.can_generate = True
        self._client = None
        self._models: list = []
        self._current_idx = 0
        self._exhausted_models: dict = {}
        self._model_cooldown_sec = MODEL_COOLDOWN_SEC
        self._forced_model = model
        self.rate_limiter = RateLimiter(max_requests_per_second=0.3)
        if SPRUTDOCK_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=SPRUTDOCK_API_KEY,
                    base_url=OPENAI_BASE_URL,
                    default_headers={"X-Title": "pdf-translator"},
                    timeout=API_TIMEOUT,
                    max_retries=0,
                )
                self._discover_free_models()
            except Exception:
                pass

    def _discover_free_models(self):
        if not self._client:
            return
        try:
            resp = self._client.models.list()
            available = [m.id for m in resp.data] if resp and resp.data else []

            if self._forced_model:
                if not available or self._forced_model in available:
                    self._models = [self._forced_model]
                    logger.info(f"SprutDock: принудительно используется модель {self._forced_model}")
                else:
                    logger.warning(f"SprutDock: запрошенная модель {self._forced_model} недоступна. Доступные: {available}")
                    self._models = available or [DEFAULT_MODEL]
            elif DEFAULT_MODEL in available:
                self._models = [DEFAULT_MODEL]
            elif available:
                free = sorted([m for m in available if "free" in m.lower()],
                              key=lambda x: (DEFAULT_MODEL.split("/")[0] in x, x), reverse=True)
                self._models = free or [DEFAULT_MODEL]
            else:
                self._models = [DEFAULT_MODEL]
            logger.info(f"SprutDock: доступно моделей: {len(self._models)}")
        except Exception as e:
            logger.warning(f"SprutDock: не удалось получить список моделей ({e}), используется {DEFAULT_MODEL}")
            self._models = [self._forced_model or DEFAULT_MODEL]

    def _is_available(self, model: str) -> bool:
        if model not in self._exhausted_models:
            return True
        if time.time() - self._exhausted_models[model] > self._model_cooldown_sec:
            del self._exhausted_models[model]
            return True
        return False

    def _next_model(self) -> Optional[str]:
        for _ in range(len(self._models)):
            model = self._models[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self._models)
            if self._is_available(model):
                return model
        return None

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        if not self._client:
            return None
        attempts = 0
        max_attempts = len(self._models) * 3
        while attempts < max_attempts:
            if STOP_EVENT.is_set():
                return None
            model = self._next_model()
            if not model:
                break
            self.rate_limiter.acquire()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _translation_system_prompt("English", self.target_lang, glossary)},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3,
                )
                content = resp.choices[0].message.content
                if content:
                    return content.strip()
                return None
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'rate' in err_str or 'limit' in err_str:
                    self._exhausted_models[model] = time.time()
                    _interruptible_sleep(5)
                    attempts += 1
                    continue
                elif '502' in err_str or '503' in err_str or '504' in err_str or 'bad gateway' in err_str or 'server error' in err_str:
                    logger.warning(f"   SprutDock: {model} -> {type(e).__name__}: {str(e)[:200]}, повтор...")
                    _interruptible_sleep(3)
                    attempts += 1
                    continue
                logger.warning(f"   SprutDock translate: {type(e).__name__}: {str(e)[:200]}")
                attempts += 1
        return None

    def generate(self, prompt: str) -> Optional[str]:
        if not self._client:
            return None
        for attempt in range(3):
            if STOP_EVENT.is_set():
                return None
            model = self._next_model()
            if not model:
                model = self._models[0] if self._models else None
            if not model:
                return None
            self.rate_limiter.acquire()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else None
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or '502' in err_str or '503' in err_str or '504' in err_str or 'bad gateway' in err_str:
                    logger.warning(f"   SprutDock generate: {type(e).__name__}: {str(e)[:200]}, повтор {attempt+1}/3...")
                    _interruptible_sleep(3)
                    continue
                logger.warning(f"   SprutDock generate: {type(e).__name__}: {str(e)[:200]}")
                return None
        return None