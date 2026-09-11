"""Google Translate backend for PDF Translator."""

import random
from typing import Optional, Dict

import requests

from .logging_setup import logger
from .rate_limiter import RateLimiter, _make_session
from .utils import _interruptible_sleep, STOP_EVENT


class GoogleTranslator:
    def __init__(self, target_lang: str):
        self.name = "Google"
        self.target_lang = target_lang
        self.can_generate = False
        self.session = _make_session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.rate_limiter = RateLimiter(max_requests_per_second=0.5)
        self._rate_limit_count = 0
        self._429_max_retries = 8

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        self.rate_limiter.acquire()
        attempt = 0
        while True:
            try:
                resp = self.session.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={"client": "gtx", "sl": "auto", "tl": self.target_lang, "dt": "t",
                            "ie": "UTF-8", "oe": "UTF-8", "q": text},
                    timeout=120
                )
                if resp.status_code == 429:
                    raise requests.exceptions.HTTPError("429 Client Error", response=resp)
                resp.raise_for_status()
                data = resp.json()
                parts = [p[0] for p in data[0] if p[0]]
                result = " ".join(parts).strip()
                self._rate_limit_count = 0
                return result if result else None
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    attempt += 1
                    self._rate_limit_count += 1
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = min(10.0 * (2 ** (self._rate_limit_count - 1)), 300.0) + random.uniform(0, 2)
                    logger.warning(f"   Google Translate: 429, пауза {delay:.0f} сек (попытка {attempt}/{self._429_max_retries})")
                    _interruptible_sleep(delay)
                    if STOP_EVENT.is_set() or attempt >= self._429_max_retries:
                        self._rate_limit_count = 0
                        return None
                    continue
                else:
                    status = e.response.status_code if e.response is not None else "?"
                    logger.warning(f"   Google Translate: HTTP {status}")
                    return None
            except Exception as e:
                logger.warning(f"   Google Translate: {type(e).__name__}")
                return None

    def generate(self, prompt: str) -> Optional[str]:
        return None