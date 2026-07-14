# translation_engine.py
import threading
import time
import random
import os
import requests
from typing import Optional, List
from requests.adapters import HTTPAdapter


class RateLimiter:
    def __init__(self, max_requests_per_second: float = 5.0):
        self.rate = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_time = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            sleep_time = self.min_interval - (now - self.last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.last_time = time.monotonic()


def _make_session(max_workers: int = 8) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers * 2)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s


# =========================================================
# Google Translate (HTTP)
# =========================================================
class GoogleTranslator:
    def __init__(self, target_lang: str):
        self.name = "Google"
        self.target_lang = target_lang
        self.session = _make_session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self.rate_limiter = RateLimiter(max_requests_per_second=3)

    def translate(self, text: str) -> Optional[str]:
        self.rate_limiter.acquire()
        try:
            resp = self.session.get(
                "https://translate.googleapis.com/translate_a/single",
                params={
                    "client": "gtx", "sl": "auto", "tl": self.target_lang,
                    "dt": "t", "ie": "UTF-8", "oe": "UTF-8", "q": text
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            parts = [p[0] for p in data[0] if p[0]]
            result = " ".join(parts).strip()
            return result if result else None
        except Exception as e:
            print(f"Google error: {e}")
            return None


# =========================================================
# OpenRouter (OpenAI-compatible)
# =========================================================
_OPENROUTER_FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "cohere/north-mini-code:free",
]

_or_last_request = 0.0
_or_lock = threading.Lock()


class OpenRouterTranslator:
    def __init__(self, target_lang: str, api_key: str,
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.name = "OpenRouter"
        self.target_lang = target_lang
        self._client = None
        self._models: List[str] = []
        self._current_idx = 0
        self._exhausted_models: dict[str, float] = {}
        self._model_cooldown_sec = 60
        self.rate_limiter = RateLimiter(max_requests_per_second=1)
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"X-Title": "pdf-translator"},
                timeout=120,
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
            self._models = sorted(
                [m.id for m in resp.data if "free" in m.id.lower()],
                key=lambda x: ("openrouter/" in x, x),
                reverse=True,
            )
            if not self._models:
                self._models = list(_OPENROUTER_FALLBACK_MODELS[:3])
        except Exception:
            self._models = list(_OPENROUTER_FALLBACK_MODELS[:3])

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

    def translate(self, text: str) -> Optional[str]:
        if not self._client:
            return None
        attempts = 0
        while attempts < len(self._models):
            model = self._next_model()
            if not model:
                break
            self.rate_limiter.acquire()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"Translate to {self.target_lang}. Return only translation, no explanations."},
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
                    time.sleep(5)
                    attempts += 1
                    continue
                attempts += 1
                continue
        return None


# =========================================================
# LlamaCpp (local server)
# =========================================================
class LlamaCppTranslator:
    def __init__(self, target_lang: str, model: str = "translate",
                 api_base: str = "http://localhost:8080/v1"):
        self.target_lang = target_lang
        self.name = "LlamaCpp"
        self.model = model
        self.api_base = api_base.rstrip("/")
        self._server_running = False
        self.rate_limiter = RateLimiter(max_requests_per_second=5)
        self._ensure_server()

    def _check_server(self) -> bool:
        try:
            resp = requests.get(f"{self.api_base}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                self._server_running = True
                if self.model not in models and models:
                    self.model = models[0]
                return True
        except Exception:
            pass
        return False

    def _ensure_server(self):
        if self._check_server():
            return
        try:
            import subprocess
            subprocess.Popen(
                ["/home/ad/bin/start_llama.sh"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            return
        for _ in range(15):
            time.sleep(2)
            if self._check_server():
                return

    def translate(self, text: str) -> Optional[str]:
        if not self._server_running:
            self._ensure_server()
        self.rate_limiter.acquire()
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": f"Translate to {self.target_lang}. Return only translation, no explanations."},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                    "stream": False,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content:
                    return content.strip()
        except Exception:
            pass
        return None


# =========================================================
# Gemini (Google AI via OpenAI-compatible API)
# =========================================================
import openai

class GeminiTranslator:
    def __init__(
        self,
        target_lang: str,
        api_key: str,
        model: str = "gemini-2.5-flash",
    ):
        self.name = "Gemini"
        self.target_lang = target_lang
        self.model = model

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=60,
        )

        self.rate_limiter = RateLimiter(max_requests_per_second=2)

    def translate(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return ""

        self.rate_limiter.acquire()

        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional translator.\n"
                                f"Translate the user's text into {self.target_lang}.\n\n"
                                "Rules:\n"
                                "- Return ONLY the translated text.\n"
                                "- Preserve formatting.\n"
                                "- Do not explain.\n"
                                "- Do not add quotation marks.\n"
                                "- Do not add comments.\n"
                                f"- If the text is already in {self.target_lang}, return it unchanged."
                            ),
                        },
                        {
                            "role": "user",
                            "content": text,
                        },
                    ],
                    temperature=0.3,
                )

                if not response.choices:
                    print("Gemini returned an empty response.")
                    return None

                message = response.choices[0].message

                if not message or not message.content:
                    print("Gemini returned no translation.")
                    return None

                return message.content.strip()

            except Exception as e:
                err = str(e).lower()

                # Повторяем запрос при превышении лимитов
                if any(x in err for x in (
                    "429",
                    "rate limit",
                    "rate_limit",
                    "quota",
                    "resource exhausted",
                    "too many requests",
                )):
                    if attempt < max_retries - 1:
                        delay = min(60, 2 ** attempt + random.uniform(0, 1))
                        print(f"Gemini rate limit. Retry in {delay:.1f} sec...")
                        time.sleep(delay)
                        continue

                # Повторяем запрос при временных ошибках сервера
                if any(x in err for x in (
                    "500",
                    "502",
                    "503",
                    "504",
                    "internal error",
                    "service unavailable",
                )):
                    if attempt < max_retries - 1:
                        delay = min(30, 2 ** attempt + random.uniform(0, 1))
                        print(f"Gemini temporary server error. Retry in {delay:.1f} sec...")
                        time.sleep(delay)
                        continue

                print(f"Gemini error: {e}")
                return None

        return None