"""Translation utility functions for PDF Translator."""

import re
import json
from typing import Optional, Dict, Any, Tuple

from .logging_setup import validation_logger

REF_PATTERN = re.compile(r'\[(?:[0-9][0-9,\s;\-–—]*|[A-Za-z][A-Za-z0-9-]*)\]')

SYSTEM_TRANSLATE_PROMPT = (
    "Translate from {source_lang} to {lang}. "
    "Return ONLY the translation. No explanations, no source text, no Markdown. "
    "Preserve paragraph boundaries and protected tokens."
)

_CYRILLIC_LANGS = {"ru", "uk", "bg", "sr", "be", "mk", "kk", "uz", "ky", "mn"}
_CYRILLIC_RE = re.compile(r'[а-яё]', re.I)
_LANG_NAMES = {
    "ru": "Russian", "en": "English", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
    "ar": "Arabic", "hi": "Hindi", "uk": "Ukrainian", "bg": "Bulgarian",
}


def _translation_system_prompt(source_lang: str, lang: str, glossary: Optional[Dict[str, str]] = None) -> str:
    prompt = SYSTEM_TRANSLATE_PROMPT.format(
        source_lang=source_lang or "the source language",
        lang=_LANG_NAMES.get(lang, lang)
    )
    if glossary:
        terms = "\n".join(f"- {k} => {v}" for k, v in glossary.items())
        prompt += "\n\nTerminology memory. Use these translations when the corresponding source term occurs. Do not output this list:\n" + terms
    return prompt


def _strip_llm_wrappers(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^\s*```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.I)
    marker_line = re.compile(r"^\s*(?:<<<\s*(?:END_)?[A-Z_][A-Z0-9_]*\s*>>>|\[\/?(?:END_)?(?:CTX|SRC|OUT)\])\s*$", re.I)
    lines = text.splitlines()
    while lines and marker_line.match(lines[0]):
        lines.pop(0)
    while lines and marker_line.match(lines[-1]):
        lines.pop()
    text = "\n".join(lines).strip()
    text = re.sub(r"<<<\s*(?:END_)?[A-Z_][A-Z0-9_]*\s*>>>", "", text, flags=re.I)
    text = re.sub(r"\[\/?(?:END_)?(?:CTX|SRC|OUT)\]", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    lines = text.splitlines()
    if lines:
        first = lines[0].strip().lower()
        preambles = ("translation:", "translated text:", "here is the translation:", "here's the translation:",
                     "перевод:", "вот перевод:")
        if first in preambles:
            lines = lines[1:]
    return "\n".join(lines).strip()


def _norm_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _extract_numbers(text: str) -> list:
    pattern = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?|\.\d+)(?:\s*[%‰])?(?!\w)")
    return pattern.findall(text or "")


_FORMULA_TOKEN_RE = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"[A-Za-zА-Яа-я](?:[_^][\d{}()+\-])?"  # variable with optional subscript/superscript
    r"|"
    r"[∫∑∏∂√∞∇±×÷≠≤≥≈∝∈∉⊂⊃∪∩∧∨¬αβγδεζηθικλμνξοπρστυφχψωΩ]"
    r")"
    r"(?:\s*[=<>≤≥≥≠±+*/÷^]\s*(?:\([^)]+\)|\{[^}]+\}|[A-Za-zА-Яа-я0-9∫∑∏∂√∞∇±×÷≠≤≥≈∝]+(?:[_^][\d{}()+\-])*))*",
    re.UNICODE,
)
_MATH_EXPR_RE = re.compile(
    r"(?:"
    r"[A-Za-zА-Яа-я0-9∫∑∏∂√∞∇±×÷≠≤≥≈∝](?:[_^][\d{}()+\-])*"
    r"\s*[=<>≤≥≥≠±+*/÷^]\s*"
    r"(?:\([^)]+\)|\{[^}]+\}|[A-Za-zА-Яа-я0-9∫∑∏∂√∞∇±×÷≠≤≥≈∝](?:[_^][\d{}()+\-])*|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"
    r")"
    r"|"
    r"(?:"
    r"[∫∑∏∂√∞∇]\s*(?:\([^)]+\)|\{[^}]+\}|[A-Za-zА-Яа-я0-9](?:[_^][\d{}()+\-])*)"
    r")",
    re.UNICODE,
)
_MATH_EXPR_RE2 = re.compile(
    r"(?:"
    r"(?:\d+(?:\.\d+)?)\s*[/÷]\s*(?:\d+(?:\.\d+)?|[A-Za-zА-Яа-я](?:[_^][\d{}()+\-])*)"
    r"|"
    r"[A-Za-zА-Яа-я](?:[_^][\d{}()+\-])*\s*[/÷]\s*[A-Za-zА-Яа-я0-9](?:[_^][\d{}()+\-])*"
    r"|"
    r"(?:\([^)]+\)|\{[^}]+\})\s*[/÷]\s*(?:\([^)]+\)|\{[^}]+\}|[A-Za-zА-Яа-я0-9](?:[_^][\d{}()+\-])*|[0-9]+(?:\.\d+)?)"
    r")",
    re.UNICODE,
)
_MATH_PAREN_RE = re.compile(
    r"(?:"
    r"[(\[]\s*[A-Za-zА-Яа-я0-9∫∑∏∂√∞∇±×÷≠≤≥≈∝](?:[_^][\d{}()+\-])*\s*(?:[=<>≤≥≥≠±+*/÷^]\s*[A-Za-zА-Яа-я0-9∫∑∏∂√∞∇±×÷≠≤≥≈∝](?:[_^][\d{}()+\-])*|[0-9]+(?:\.\d+)?)\s*[)\]]"
    r"|"
    r"[(\[]\s*[0-9]+(?:\.\d+)?\s*(?:[,;]\s*[0-9]+(?:\.\d+)?)\s*[)\]]"
    r")",
    re.UNICODE,
)
_SPECIAL_MATH_CHARS = set("∫∑∏∂√∞∇αβγδεζηθικλμνξοπρστυφχψωΩ")


def _extract_protected_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    protected: Dict[str, str] = {}
    counter = 0

    def add_value(value: str, prefix: str = "P") -> str:
        nonlocal counter
        if not value or not value.strip():
            return value
        key = f"[[{prefix}_{counter:04d}]]"
        counter += 1
        protected[key] = value
        return f" {key} "

    result = strip_html_tags(text)

    # Защита HTML ссылок <a href="...">...</a>
    html_link_pattern = re.compile(r"<a\s+[^>]*href=['\"][^'\"]+['\"][^>]*>.*?</a>", re.I | re.DOTALL)
    for m in html_link_pattern.finditer(result):
        value = m.group(0).strip()
        result = result[:m.start()] + add_value(value, "LINK") + result[m.end():]

    # Защита атрибутов href="..."
    href_pattern = re.compile(r"href=['\"][^'\"]+['\"]", re.I)
    for m in href_pattern.finditer(result):
        value = m.group(0).strip()
        result = result[:m.start()] + add_value(value, "LINK") + result[m.end():]

    # Защита всей академической цитаты
    surname = r"[A-ZА-Я][A-Za-zА-Яа-я'’\-]+"
    citation_pattern = re.compile(
        rf"\([^\n(){{}}]{{0,220}}(?:19|20)\d{{2}}[^\n(){{}}]{{0,40}}\)"
        rf"|\b{surname}(?:\s*,\s*{surname})*(?:\s+et\s+al\.?)?\s*\((?:19|20)\d{{2}}[^)]*\)",
        re.UNICODE | re.I,
    )
    spans = []
    for cm in citation_pattern.finditer(result):
        value = cm.group(0).strip()
        spans.append((cm.start(), cm.end(), value))
    for a, b, value in sorted(set(spans), reverse=True):
        result = result[:a] + add_value(value, "CITE") + result[b:]

    # Защита математических выражений (x = 0.05, p < 0.01, ∫ f(x) dx и т.д.)
    math_spans = []
    for m in _MATH_EXPR_RE.finditer(result):
        value = m.group(0).strip()
        if len(value) > 3:
            math_spans.append((m.start(), m.end(), value))
    for m in _MATH_EXPR_RE2.finditer(result):
        value = m.group(0).strip()
        if len(value) > 3:
            math_spans.append((m.start(), m.end(), value))
    for m in _MATH_PAREN_RE.finditer(result):
        value = m.group(0).strip()
        if len(value) > 3:
            math_spans.append((m.start(), m.end(), value))
    for a, b, value in sorted(set(math_spans), reverse=True):
        result = result[:a] + add_value(value, "MATH") + result[b:]

    # Защита отдельных токенов с спецсимволами (например: h_B_n, x^2, ∂/∂x)
    math_spans2 = []
    for m in _FORMULA_TOKEN_RE.finditer(result):
        value = m.group(0).strip()
        if any(c in _SPECIAL_MATH_CHARS for c in value) or '_' in value or '^' in value:
            if len(value) > 1:
                math_spans2.append((m.start(), m.end(), value))
    for a, b, value in sorted(set(math_spans2), reverse=True):
        result = result[:a] + add_value(value, "MATH") + result[b:]

    patterns = [
        re.compile(r"\[\[REF\d+\]\]"),
        REF_PATTERN,
        re.compile(r"https?://[^\s<>\]\)]+", re.I),
        re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I),
        re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9_]*(?:\s*[=<>±]\s*[^,.;\n]+)?"),
    ]

    def add(match):
        value = match.group(0)
        if not value.strip():
            return value
        if re.fullmatch(r"[A-Za-z]{1,20}", value):
            return value
        return add_value(value)

    for pat in patterns:
        result = pat.sub(add, result)
    return result, protected


def _restore_protected(text: str, protected: Dict[str, str]) -> str:
    result = text or ""
    for token, original in protected.items():
        pattern = re.escape(token).replace(r"\[\[", r"\[\[\s*").replace(r"\]\]", r"\s*\]\]")
        result = re.sub(pattern, lambda m: original, result, flags=re.I | re.DOTALL)
    result = re.sub(r"\[\[\s*(?:AUTHOR|P|REF|CITE|LINK|MATH)_\d+\s*\]\]", "", result, flags=re.I | re.DOTALL)
    return result


def strip_html_tags(text: str) -> str:
    """Remove HTML formatting tags (<b>, <i>, etc.) from text, keeping inner content."""
    if not text:
        return text
    result = re.sub(r"</?(?:b|i|u|em|strong|small|sub|sup|span|font)[^>]*>", "", text, flags=re.I)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _repetition_ratio(text: str, n: int = 6) -> float:
    words = re.findall(r"\w+", (text or "").casefold(), flags=re.UNICODE)
    if len(words) < n * 2:
        return 0.0
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


def _looks_like_meta_response(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    lines = [x.strip().casefold() for x in t.splitlines() if x.strip()]
    if not lines:
        return False
    bad_prefixes = ("analysis:", "reasoning:", "chain of thought:", "thoughts:", "translation note:",
                    "translator note:", "here is the translation:", "вот перевод:", "анализ:",
                    "рассуждение:", "объяснение:", "я не могу перевести", "не могу перевести")
    if any(line.startswith(bad_prefixes) for line in lines[:2]):
        return True
    meta = re.search(r"\b(i cannot|i can't|i am unable|as an ai|as a language model)\b", t, re.I)
    return bool(meta)


def validate_translation(source: str, candidate: str, target_lang: str,
                         protected: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, Any]]:
    source = (source or "").strip()
    candidate = _strip_llm_wrappers(candidate)
    info: Dict[str, Any] = {"echo": False, "too_short": False, "too_long": False,
                            "repetition": False, "numbers_changed": False,
                            "protected_missing": [], "language_mismatch": False, "meta_response": False}
    if not candidate:
        return False, info
    info["meta_response"] = _looks_like_meta_response(candidate)

    src_norm = _norm_for_compare(source)
    cand_norm = _norm_for_compare(candidate)

    if len(source) >= 20:
        if cand_norm == src_norm:
            info["echo"] = True
        elif src_norm in cand_norm and len(src_norm) > 100 and len(src_norm) > len(cand_norm) * 0.8:
            info["echo"] = True

    if len(source) >= 150 and len(candidate) < max(30, int(len(source) * 0.15)):
        info["too_short"] = True
    if len(source) >= 100 and len(candidate) > int(len(source) * 3.2):
        info["too_long"] = True

    if _repetition_ratio(candidate) > 0.30:
        info["repetition"] = True

    def _norm_num(x: str):
        x = x.strip().replace(" ", "").replace("", "-")
        x = re.sub(r"[%‰]$", "", x)
        try:
            return ("num", float(x.replace(",", ".")))
        except ValueError:
            try:
                return ("num", float(x.replace(",", "")))
            except ValueError:
                return ("txt", x.casefold())

    src_nums = sorted(_norm_num(x) for x in _extract_numbers(source))
    cand_nums = sorted(_norm_num(x) for x in _extract_numbers(candidate))
    if src_nums != cand_nums:
        info["numbers_changed"] = True

    if protected:
        info["protected_missing"] = [token for token in protected if token not in candidate]

    if target_lang in _CYRILLIC_LANGS and len(candidate) > 30:
        letters = [c for c in candidate if c.isalpha()]
        cyr = sum(1 for c in letters if _CYRILLIC_RE.match(c))
        if letters and cyr / len(letters) < 0.12:
            info["language_mismatch"] = True

    hard = bool(info["echo"] or info["too_short"] or info["too_long"] or info["meta_response"] or not candidate)
    if hard:
        reasons = [k for k, v in info.items() if v and k not in ("protected_missing",)]
        validation_logger.info(json.dumps({
            "result": "rejected",
            "reasons": reasons,
            "source_len": len(source),
            "candidate_len": len(candidate) if candidate else 0,
            "source_preview": source[:120],
        }, ensure_ascii=False))
    return (not hard), info