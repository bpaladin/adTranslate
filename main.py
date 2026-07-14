#!/usr/bin/env python3
"""
PDF Translator — CLI wrapper
Весь функционал в modules/ (pipeline, translators, cache, html).
"""

import sys
import os
import logging
import warnings
import argparse

# =========================================================
# НАСТРОЙКА ЛОГГИРОВАНИЯ
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

SCRIPT_VERSION = "0.9"

# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description=f"PDF Translator v{SCRIPT_VERSION}")
    parser.add_argument("input", help="Входной PDF файл")
    parser.add_argument("output", help="Выходной HTML файл")
    parser.add_argument("-l", "--lang", default="ru",
                        help="Целевой язык (по умолчанию: ru)")
    parser.add_argument("-t", "--translator", default="google",
                        choices=["google", "openrouter", "llama", "gemini"],
                        help="Переводчик (по умолчанию: google)")
    parser.add_argument("--api-key", default=None,
                        help="API ключ (или переменная окружения)")
    parser.add_argument("--llama-model", default="gemma4",
                        help="Модель llama.cpp (по умолчанию: gemma4)")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1",
                        help="URL llama.cpp сервера")
    parser.add_argument("--gemini-model", default="gemini-flash-latest",
                        help="Модель Gemini (по умолчанию: gemini-flash-latest)")
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 1) * 2),
                        help="Число параллельных воркеров")
    parser.add_argument("--task-timeout", type=int, default=600,
                        help="Таймаут на перевод (сек)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Тихий режим")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Подробный вывод")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info(f"PDF TRANSLATOR v{SCRIPT_VERSION} — {args.translator.upper()}")
    logger.info("=" * 60)

    try:
        from modules.pipeline import process_pdf

        result = process_pdf(
            pdf_path=args.input,
            output_html=args.output,
            lang=args.lang,
            translator_type=args.translator,
            api_key=args.api_key,
            llama_model=args.llama_model,
            llama_url=args.llama_url,
            gemini_model=args.gemini_model,
            max_workers=args.workers,
            timeout=args.task_timeout,
            quiet=args.quiet,
        )

        stats = result["stats"]
        logger.info("=" * 60)
        logger.info("📊 СТАТИСТИКА:")
        logger.info(f"   ✓ Переведено: {stats['success']}")
        logger.info(f"   ⚡Из кэша:   {stats['cached']}")
        logger.info(f"   ⊘ Пропущено:  {stats['skipped']}")
        logger.info(f"   ✗ Ошибок:     {stats['failed']}")
        logger.info("=" * 60)
        logger.info(f"✅ Готово: {args.output}")

    except ImportError as e:
        logger.error(f"Ошибка импорта модулей: {e}")
        logger.error("Убедитесь, что установлены зависимости: pip install pymupdf openai requests jinja2")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("⚠️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
