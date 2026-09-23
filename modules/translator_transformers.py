"""CTranslate2 (Helsinki-NLP Opus-MT) translator backend for PDF Translator.

Рантайм: ctranslate2.
Токенизатор: transformers + sentencepiece (AutoTokenizer с HF-хаба).
Модель по умолчанию: Helsinki-NLP/opus-mt-en-ru.

Фиксированные параметры инференса (по ТЗ):
    device="cpu", compute_type="int8",
    inter_threads=1, intra_threads=0.

Качество перевода управляется пресетами ``NMT_QUALITY``:
    fast       — beam=2, patience=0.7  (превью, черновики)
    balanced   — beam=4, patience=1.0  (по умолчанию)
    quality    — beam=6, patience=1.0  (рекомендуется для PDF)
    max        — beam=8, patience=1.2  (максимум качества, медленно)

Любой параметр можно переопределить через env (NMT_BEAM_SIZE, NMT_PATIENCE,
NMT_LENGTH_PENALTY, NMT_REPETITION_PENALTY, NMT_NO_REPEAT_NGRAM,
NMT_MAX_BATCH_SIZE, NMT_MAX_INPUT_TOKENS, NMT_MAX_CHARS) — env имеет
приоритет над пресетом.

HF-модель автоматически конвертируется в формат CT2 один раз
(``ct2-transformers-converter --quantization int8``) и кэшируется в
``~/.cache/adTranslate/ct2`` (переопределяется через ``CT2_MODEL_DIR``).
Можно передать и готовый локальный каталог CT2 (с ``model.bin``) —
тогда конвертация пропускается, а HF-id токенизатора читается из
``hf_id.txt`` рядом (или берётся дефолт под целевой язык).
"""

import os
import re
import subprocess
import threading
from typing import Optional, Dict

from .logging_setup import logger
from .utils import STOP_EVENT, split_by_sentences

# Дефолтные модели под целевой язык (источник — английский, основной кейс PDF).
# Значение — HF-id: из него берётся токенизатор и (при нужды) делается CT2-конвертация.
DEFAULT_MODELS = {
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "en": "Helsinki-NLP/opus-mt-ru-en",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
}

# ── Пресеты качества перевода ─────────────────────────────────────────────
# Значения — разумные дефолты; env-переменные NMT_* переопределяют их.
QUALITY_PRESETS: Dict[str, Dict[str, float]] = {
    "fast": {
        "beam_size": 2,
        "patience": 0.7,
        "length_penalty": 1.0,
        "repetition_penalty": 1.02,
        "no_repeat_ngram_size": 4,
        "max_batch_size": 32,
    },
    "balanced": {
        "beam_size": 4,
        "patience": 1.0,
        "length_penalty": 1.0,
        "repetition_penalty": 1.02,
        "no_repeat_ngram_size": 4,
        "max_batch_size": 16,
    },
    "quality": {
        "beam_size": 6,
        "patience": 1.0,
        "length_penalty": 1.0,
        "repetition_penalty": 1.02,
        "no_repeat_ngram_size": 4,
        "max_batch_size": 16,
    },
    "max": {
        "beam_size": 8,
        "patience": 1.2,
        "length_penalty": 1.0,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 4,
        "max_batch_size": 16,
    },
}

_DEFAULT_QUALITY = os.getenv("NMT_QUALITY", "quality").strip().lower()
if _DEFAULT_QUALITY not in QUALITY_PRESETS:
    _DEFAULT_QUALITY = "max"


def _resolve_quality_params(quality: Optional[str]) -> Dict[str, float]:
    """Собирает финальные параметры: пресет → env-overrides."""
    key = (quality or _DEFAULT_QUALITY or "balanced").strip().lower()
    preset = dict(QUALITY_PRESETS.get(key, QUALITY_PRESETS["balanced"]))
    # env-overrides имеют приоритет над пресетом.
    env_map = {
        "beam_size": ("NMT_BEAM_SIZE", int),
        "patience": ("NMT_PATIENCE", float),
        "length_penalty": ("NMT_LENGTH_PENALTY", float),
        "repetition_penalty": ("NMT_REPETITION_PENALTY", float),
        "no_repeat_ngram_size": ("NMT_NO_REPEAT_NGRAM", int),
        "max_batch_size": ("NMT_MAX_BATCH_SIZE", int),
    }
    for param, (env_name, caster) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        try:
            preset[param] = caster(raw)
        except (TypeError, ValueError):
            logger.warning(f"   CTranslate2: некорректное значение {env_name}={raw!r}, игнорирую")
    return preset


# ── Фиксированные рантайм-параметры (по ТЗ) ───────────────────────────────
_CT2_DEVICE = "cpu"
_CT2_COMPUTE_TYPE = "int8"
_CT2_INTER_THREADS = 1
_CT2_INTRA_THREADS = 0

# Opus-MT обучен на 512 токенов — больше делать max_input_length бессмысленно.
_CT2_MAX_INPUT_LENGTH = int(os.getenv("NMT_MAX_INPUT_TOKENS", "512"))

# Opus-MT принимает ~512 токенов; для латиницы ~4 симв./токен, для кириллицы ~2.
_MAX_CHARS_PER_PASS = int(os.getenv("NMT_MAX_CHARS", "1200"))
# Источники с кириллицей/сложной морфологией режем сильнее.
_TIGHT_SOURCE_LANGS = frozenset({"ru", "uk", "bg", "sr", "mk"})


def _split_for_nmt(text: str, max_chars: int = _MAX_CHARS_PER_PASS):
    # min_chunk_chars=0: склейка хвостов не нужна, как раньше.
    return split_by_sentences(text, max_chars, 0)


def _split_paragraphs(text: str):
    """Делит текст на абзацы по пустым строкам, сохраняя непустые."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_for_nmt_with_paragraphs(text: str, max_chars: int):
    """Возвращает список (paragraph_index, sentence) — чтобы собрать абзацы обратно."""
    items = []
    for pi, para in enumerate(_split_paragraphs(text)):
        for sent in split_by_sentences(para, max_chars, 0):
            s = sent.strip()
            if s:
                items.append((pi, s))
    return items


def _postprocess_target(text: str, lang: str) -> str:
    """Лёгкая нормализация вывода NMT (убирает типичные артефакты)."""
    if not text:
        return text
    text = re.sub(r"[ \t]+", " ", text).strip()
    # Пробел перед знаком препинания — артефакт NMT.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Пробел после открывающей скобки/кавычки.
    text = re.sub(r"([(\[«„])\s+", r"\1", text)
    # Гарантируем пробел после знака препинания, если следом буква.
    text = re.sub(r"([,.;:!?])(?=[A-Za-zА-Яа-яЁё])", r"\1 ", text)
    if lang == "ru":
        text = re.sub(r"\s*—\s*", " — ", text)
        text = text.replace("...", "…")
    return text


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)


def _default_ct2_dir(hf_id: str) -> str:
    base = os.getenv("CT2_MODEL_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "adTranslate", "ct2")
    return os.path.join(base, f"{_safe_name(hf_id)}_int8")


def _convert_hf_to_ct2(hf_id: str, out_dir: str) -> None:
    """Одноразовая конвертация HF-модели в формат CTranslate2 (int8)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ct2-transformers-converter",
           "--model", hf_id,
           "--output_dir", out_dir,
           "--quantization", "int8",
           "--force"]
    logger.info(f"   CTranslate2: конвертация {hf_id} -> {out_dir} (int8)...")
    subprocess.run(cmd, check=True)
    with open(os.path.join(out_dir, "hf_id.txt"), "w", encoding="utf-8") as f:
        f.write(hf_id)


class TransformersTranslator:
    """Локальный NMT-переводчик на CTranslate2.

    Управление качеством:
        * ``quality="fast"|"balanced"|"quality"|"max"`` в конструкторе
        * либо env ``NMT_QUALITY`` (default: ``quality``)
        * любой параметр — env ``NMT_*`` (см. ``QUALITY_PRESETS``)

    Дополнительно:
        * сохраняется абзацная структура (``\\n\\n``);
        * batch сортируется по длине для лучшего packing;
        * пост-обработка убирает «висячие» пробелы перед пунктуацией.
    """

    def __init__(self, target_lang: str, model: Optional[str] = None,
                 source_lang: str = "en", max_length: int = 512,
                 quality: Optional[str] = None):
        self.target_lang = (target_lang or "ru").lower()
        self.name = "Transformers"
        self.can_generate = False
        self.source_lang = source_lang
        self.max_length = max_length
        # model: HF-id для конвертации ИЛИ готовый локальный каталог CT2.
        self.model_name = model or DEFAULT_MODELS.get(
            self.target_lang, f"Helsinki-NLP/opus-mt-en-{self.target_lang}")

        # ── Параметры качества ──
        self.quality = (quality or _DEFAULT_QUALITY or "balanced").strip().lower()
        if self.quality not in QUALITY_PRESETS:
            logger.warning(
                f"   CTranslate2: неизвестный quality={self.quality!r}, "
                f"использую 'balanced'")
            self.quality = "balanced"
        params = _resolve_quality_params(self.quality)
        self.beam_size: int = int(params["beam_size"])
        self.patience: float = float(params["patience"])
        self.length_penalty: float = float(params["length_penalty"])
        self.repetition_penalty: float = float(params["repetition_penalty"])
        self.no_repeat_ngram_size: int = int(params["no_repeat_ngram_size"])
        self.max_batch_size: int = int(params["max_batch_size"])

        self._tokenizer = None
        self._translator = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    # ── Публичный доступ к настройкам качества ────────────────────────────
    def quality_info(self) -> Dict[str, object]:
        return {
            "quality": self.quality,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "max_batch_size": self.max_batch_size,
            "max_input_tokens": _CT2_MAX_INPUT_LENGTH,
        }

    def _resolve_dirs(self):
        """Возвращает (ct2_dir, hf_id): где брать рантайм-модель и токенизатор."""
        if os.path.isdir(self.model_name) and os.path.exists(
                os.path.join(self.model_name, "model.bin")):
            hf_id = None
            marker = os.path.join(self.model_name, "hf_id.txt")
            if os.path.exists(marker):
                try:
                    with open(marker, encoding="utf-8") as f:
                        hf_id = f.read().strip() or None
                except Exception:
                    pass
            hf_id = hf_id or DEFAULT_MODELS.get(
                self.target_lang, f"Helsinki-NLP/opus-mt-en-{self.target_lang}")
            return self.model_name, hf_id
        hf_id = self.model_name
        return _default_ct2_dir(hf_id), hf_id

    def _ensure_loaded(self) -> bool:
        if self._translator is not None:
            return True
        if self._load_error is not None:
            return False
        with self._lock:
            if self._translator is not None:
                return True
            if self._load_error is not None:
                return False
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                self._load_error = (f"transformers/sentencepiece не установлены ({e}). "
                                    "Установите: pip install transformers sentencepiece")
                logger.error(f"   CTranslate2: {self._load_error}")
                return False
            ct2_dir, hf_id = self._resolve_dirs()
            try:
                if not os.path.exists(os.path.join(ct2_dir, "model.bin")):
                    _convert_hf_to_ct2(hf_id, ct2_dir)
                logger.info(f"   CTranslate2: загрузка {ct2_dir} "
                            f"(cpu, int8, tokenizer={hf_id}, quality={self.quality}, "
                            f"beam={self.beam_size})...")
                self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
                import ctranslate2
                self._translator = ctranslate2.Translator(
                    ct2_dir,
                    device=_CT2_DEVICE,
                    compute_type=_CT2_COMPUTE_TYPE,
                    inter_threads=_CT2_INTER_THREADS,
                    intra_threads=_CT2_INTRA_THREADS,
                )
                logger.info(f"   CTranslate2: модель загружена ({ct2_dir})")
                return True
            except ImportError as e:
                self._load_error = (f"ctranslate2 не установлен ({e}). "
                                    "Установите: pip install ctranslate2")
                logger.error(f"   CTranslate2: {self._load_error}")
                return False
            except subprocess.CalledProcessError as e:
                self._load_error = f"конвертация {hf_id} не удалась: {e}"
                logger.warning(f"   CTranslate2: {self._load_error}")
                return False
            except FileNotFoundError as e:
                self._load_error = (f"ct2-transformers-converter не найден ({e}). "
                                    "Установите: pip install ctranslate2")
                logger.error(f"   CTranslate2: {self._load_error}")
                return False
            except Exception as e:
                self._load_error = f"не удалось загрузить {ct2_dir}: {e}"
                logger.warning(f"   CTranslate2: {self._load_error}")
                return False

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        # glossary игнорируется: Opus-MT — чистый NMT без системного промпта.
        if not text or not text.strip():
            return None
        if not self._ensure_loaded():
            return None
        try:
            with self._lock:
                src = text.strip()
                max_chars = (_MAX_CHARS_PER_PASS
                             if self.source_lang not in _TIGHT_SOURCE_LANGS
                             else 800)
                items = _split_for_nmt_with_paragraphs(src, max_chars)
                if not items or STOP_EVENT.is_set():
                    return None

                # ── Сортировка по длине для лучшего batch packing в CT2 ──
                order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
                sorted_texts = [items[i][1] for i in order]
                source_batch = [
                    self._tokenizer.convert_ids_to_tokens(
                        self._tokenizer.encode(p, add_special_tokens=True))
                    for p in sorted_texts
                ]

                results = self._translator.translate_batch(
                    source_batch,
                    beam_size=self.beam_size,
                    patience=self.patience,
                    length_penalty=self.length_penalty,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    max_input_length=_CT2_MAX_INPUT_LENGTH,
                    max_batch_size=self.max_batch_size,
                    batch_type="tokens",
                    max_decoding_length=self.max_length,
                )

                # ── Рас-сортировка: возвращаем в исходный порядок ──
                decoded = [""] * len(items)
                for pos, res in zip(order, results):
                    if not res.hypotheses:
                        # Пустая гипотеза → пробуем greedy для этого куска.
                        try:
                            greedy = self._translator.translate_batch(
                                [source_batch[pos]],
                                beam_size=1,
                                max_input_length=_CT2_MAX_INPUT_LENGTH,
                                max_decoding_length=self.max_length,
                            )
                            if greedy and greedy[0].hypotheses:
                                res = greedy[0]
                            else:
                                continue
                        except Exception:
                            continue
                    ids = self._tokenizer.convert_tokens_to_ids(res.hypotheses[0])
                    out_txt = self._tokenizer.decode(ids, skip_special_tokens=True).strip()
                    if out_txt:
                        decoded[pos] = _postprocess_target(out_txt, self.target_lang)

                # ── Сборка с сохранением абзацев ──
                para_buckets: Dict[int, list] = {}
                for (pi, _), out_txt in zip(items, decoded):
                    if out_txt:
                        para_buckets.setdefault(pi, []).append(out_txt)
                if not para_buckets:
                    return None
                parts = [" ".join(para_buckets[pi]) for pi in sorted(para_buckets)]
                result = "\n\n".join(p for p in parts if p).strip()
                return result or None
        except Exception as e:
            logger.warning(f"   CTranslate2 translate: {type(e).__name__}: {str(e)[:200]}")
            return None

    def generate(self, prompt: str) -> Optional[str]:
        return None