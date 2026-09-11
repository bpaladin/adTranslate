"""LlamaCpp translator backend for PDF Translator."""

import os
import re
import time
import subprocess
from typing import Optional, Dict, List, Tuple

import requests

from .logging_setup import logger
from .rate_limiter import RateLimiter
from .translation_utils import _translation_system_prompt, _extract_protected_tokens, _strip_llm_wrappers, _restore_protected, validate_translation
from .utils import normalize_text, _interruptible_sleep, STOP_EVENT
from .config import REQUEST_TIMEOUT


def _model_matches(expected: Optional[str], actual: Optional[str]) -> bool:
    if not expected or not actual:
        return False
    e = expected.lower()
    a = actual.lower()
    if e == a:
        return True
    e_base = os.path.basename(e).lower()
    a_base = os.path.basename(a).lower()
    return any(sub in other for sub, other in ((e, a), (a, e), (e_base, a_base), (a_base, e_base)))


def _is_short_citation(text: str) -> bool:
    t = normalize_text(text)
    if not t or len(t) > 140:
        return False
    citation = re.compile(
        r"^\s*(?:\(?[A-ZА-Я][A-Za-zА-Яа-я'’\-]+(?:\s+(?:et\s+al\.?|and\s+[A-ZА-Я][A-Za-zА-Яа-я'’\-]+|&\s*[A-ZА-Я][A-Za-zА-Яа-я'’\-]+))?\s*,?\s*(?:19|20)\d{2}[a-z]?\)?|\(?(?:[A-ZА-Я][A-Za-zА-Яа-я'’\-]+(?:\s*,\s*|\s*&\s*)?)+\s*,\s*(?:19|20)\d{2}[a-z]?\)?)[.,;]?\s*$",
        re.UNICODE,
    )
    if citation.match(t):
        return True
    if re.fullmatch(r"\s*(?:\d{1,3}(?:\s*[-–—,;]\s*\d{1,3})*|\[[0-9,\-–— ]+\])\s*[.]?\s*", t):
        return True
    if len(t) <= 50 and re.fullmatch(r"(?:[A-Z][A-Za-z'’\-]+(?:\s+et\s+al\.?)?)(?:\s*\(?(?:19|20)\d{2}\)?)?\.?", t):
        return bool(re.search(r"(?:19|20)\d{2}|et\s+al", t, re.I))
    return False


class LlamaCppTranslator:
    def __init__(self, target_lang: str, api_base: str = "http://localhost:8080/v1",
                 expected_model: Optional[str] = None, auto_find: bool = True,
                 max_retries: int = 1, source_lang: str = "English"):
        self.target_lang = target_lang
        self.name = "LlamaCpp"
        self.can_generate = True
        self.max_retries = max(1, min(int(max_retries), 6))
        self.source_lang = source_lang
        self.expected_model = expected_model
        self.rate_limiter = RateLimiter(max_requests_per_second=1)
        self._loaded_model = None
        self._server_ok = False
        self.api_base = None
        self._consecutive_failures = 0
        self._last_health_check = 0.0

        if api_base:
            ok, models, found_url = self._check_server(api_base)
            if ok:
                server_model = models[0] if models else "unknown"
                if expected_model and not _model_matches(expected_model, server_model):
                    logger.warning(f"   На {found_url} загружена '{server_model}', а нужна '{expected_model}'")
                else:
                    self.api_base = found_url
                    self._loaded_model = server_model
                    self._server_ok = True
                    logger.info(f"   llama-server: {self.api_base}, модель: {self._loaded_model}")

        if not self._server_ok and auto_find:
            servers = self._find_all_servers()
            if not servers:
                raise RuntimeError("llama-server не найден. Запустите: ./start_llama.sh translate 8080")
            if expected_model is None:
                self.api_base = servers[0]["url"]
                self._loaded_model = servers[0]["model"]
                self._server_ok = True
            else:
                matched = [x for x in servers if _model_matches(expected_model, x["model"])]
                if matched:
                    self.api_base = matched[0]["url"]
                    self._loaded_model = matched[0]["model"]
                    self._server_ok = True
                else:
                    available = "\n".join(f"   {x['url']} -> {x['model']}" for x in servers)
                    raise RuntimeError(f"Модель '{expected_model}' не найдена.\n{available}")

        if not self._server_ok:
            raise RuntimeError("Не удалось подключиться ни к одному серверу.")

    @staticmethod
    def _find_all_servers() -> List[Dict[str, str]]:
        if os.name == 'nt':
            return []
        result = []
        try:
            output = subprocess.check_output(["pgrep", "-a", "llama-server"], text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                m = re.search(r"--port\s+(\d+)", line)
                port = int(m.group(1)) if m else 8080
                url = f"http://localhost:{port}/v1"
                ok, models, found_url = LlamaCppTranslator._check_server(url)
                if ok and models:
                    result.append({"url": found_url, "model": models[0]})
        except Exception:
            pass
        return result

    def _check_server(self, url: str) -> Tuple[bool, List[str], str]:
        base = url.rstrip("/")
        candidates = [base]
        if base.endswith("/v1"):
            candidates.append(base[:-3])
        else:
            candidates.append(base + "/v1")
        for b in candidates:
            try:
                resp = requests.get(f"{b}/models", timeout=(2, 5))
                if resp.status_code == 200:
                    models = [m.get("id", "") for m in resp.json().get("data", [])]
                    if models:
                        return True, models, b
            except Exception:
                pass
        return False, [], url

    def _health_check(self) -> bool:
        now = time.monotonic()
        if now - self._last_health_check < 30:
            return self._server_ok
        self._last_health_check = now
        ok, _, found_url = self._check_server(self.api_base)
        if not ok:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._server_ok = False
                logger.warning("   llama-server недоступен (3+ ошибки подряд). Требуется перезапуск.")
        else:
            self._consecutive_failures = 0
            self._server_ok = True
            if found_url != self.api_base:
                self.api_base = found_url
        return self._server_ok

    @property
    def is_healthy(self) -> bool:
        return self._server_ok and self._consecutive_failures < 3

    def _call_completion(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048,
                         timeout: int = REQUEST_TIMEOUT, stop: Optional[List[str]] = None) -> Optional[str]:
        if not self._server_ok:
            return None
        timeout = max(1, int(timeout))
        io_timeout = (10, timeout)
        self.rate_limiter.acquire()
        logger.debug(f"llama: {len(prompt)} chars, timeout={timeout}s")
        try:
            payload = {
                "model": self._loaded_model,
                "messages": [
                    {"role": "system", "content": _translation_system_prompt(self.source_lang, self.target_lang)},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens, "stream": False,
            }
            if stop:
                payload["stop"] = stop

            resp = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=io_timeout)
            if resp.status_code == 200:
                self._consecutive_failures = 0
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if content:
                        logger.debug("llama: 200 OK")
                        return content.strip()
                content = data.get("content")
                if content:
                    logger.debug("llama: 200 OK")
                    return str(content).strip()

            if resp.status_code in (404, 405, 501):
                payload2 = {"model": self._loaded_model, "prompt": prompt, "temperature": temperature,
                            "top_p": 0.9, "max_tokens": max_tokens, "stream": False}
                if stop:
                    payload2["stop"] = stop
                resp2 = requests.post(f"{self.api_base}/completions", json=payload2, timeout=io_timeout)
                if resp2.status_code == 200:
                    self._consecutive_failures = 0
                    logger.debug("llama: 200 OK (completions fallback)")
                    data = resp2.json()
                    choices = data.get("choices") or [{}]
                    content = data.get("content")
                    if not content and isinstance(choices[0], dict):
                        content = choices[0].get("text")
                    return content.strip() if content else None
            logger.warning(f"llama: HTTP {resp.status_code}")
            self._consecutive_failures += 1
            return None
        except Exception:
            self._consecutive_failures += 1
            return None

    def _translation_prompt(self, source: str, lang: str, glossary: Optional[Dict[str, str]] = None,
                            context_before: str = "", context_after: str = "", strict: bool = False) -> Tuple[str, Dict[str, str]]:
        protected, tokens = _extract_protected_tokens(source)
        sys_prompt = _translation_system_prompt(self.source_lang, lang, glossary)
        parts = [sys_prompt, ""]
        if context_before:
            parts += ["[CTX]", context_before, "[/CTX]", ""]
        parts += ["[SRC]", protected, "[/SRC]", ""]
        if context_after:
            parts += ["[CTX]", context_after, "[/CTX]", ""]
        parts += ["[OUT]"]
        return "\n".join(parts), tokens

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None,
                  context_before: str = "", context_after: str = "", strict: bool = False,
                  deadline: Optional[float] = None) -> Optional[str]:
        source = normalize_text(text)
        if not source:
            return None

        prompt, protected = self._translation_prompt(source, self.target_lang, glossary,
                                                      context_before=context_before, context_after=context_after, strict=strict)

        temperatures = (0.0, 0.1, 0.2)
        for attempt in range(self.max_retries):
            if STOP_EVENT.is_set():
                return None
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 1.0:
                return None
            req_timeout = REQUEST_TIMEOUT if remaining is None else max(1, min(REQUEST_TIMEOUT, int(remaining)))
            max_tokens = max(512, min(4096, int(len(source) / 2.0) + 512))

            raw = self._call_completion(prompt, temperature=temperatures[attempt % len(temperatures)],
                                        max_tokens=max_tokens, timeout=req_timeout)
            candidate_raw = _strip_llm_wrappers(raw)
            candidate = _restore_protected(candidate_raw, protected)
            ok, reason = validate_translation(source, candidate, self.target_lang)
            if ok:
                return candidate.strip()

            if attempt + 1 < self.max_retries:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 1.0:
                    return None
                _interruptible_sleep(min(1.5 * (2 ** attempt), 8.0))
        return None

    def generate(self, prompt: str) -> Optional[str]:
        return self._call_completion(prompt, temperature=0.2, max_tokens=1024)