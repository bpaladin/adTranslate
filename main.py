#!/usr/bin/env python3
"""
PDF Translator — единый скрипт для перевода и реферирования PDF-документов.
Поддерживает Google Translate, OpenRouter/SprutDock и локальный llama.cpp.
"""

import os
import sys
import glob
import logging
import argparse

from modules.utils import __version__, _format_time
from modules.logging_setup import logger
from modules.config import BLOCK_TIMEOUT
from modules.pipeline import process_pdf, process_pdf_raw


def interactive_mode():
    seen = set()
    pdf_files = []
    for f in glob.glob("*.[pP][dD][fF]"):
        key = os.path.normcase(f)
        if key not in seen:
            seen.add(key)
            pdf_files.append(f)
    pdf_files.sort(key=lambda x: os.path.normcase(x))
    if not pdf_files:
        print("В текущем каталоге нет PDF-файлов.")
        sys.exit(0)

    print("=" * 60)
    print(f"  PDF TRANSLATOR v{__version__}")
    print("=" * 60)
    print("\nДоступные PDF-файлы:\n")
    for i, f in enumerate(pdf_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  [{i}] {f}  ({size_mb:.1f} МБ)")
    print()

    while True:
        try:
            choice = input("Выберите файл (номер): ").strip()
            if not choice:
                continue
            idx = int(choice) - 1
            if 0 <= idx < len(pdf_files):
                break
            print(f"  Введите число от 1 до {len(pdf_files)}")
        except (ValueError, EOFError):
            print("  Некорректный ввод")

    selected_pdf = pdf_files[idx]
    print(f"\nВыбран: {selected_pdf}\n")
    print("Действия:")
    print("  [1] Перевод")
    print("  [2] Реферат")
    print("  [3] Сохранение в HTML (без перевода)")
    print()

    while True:
        try:
            action = input("Выберите действие: ").strip()
            if action in ("1", "2", "3"):
                break
            print("  Введите 1, 2 или 3")
        except (ValueError, EOFError):
            print("  Некорректный ввод")

    input_base = os.path.splitext(selected_pdf)[0]

    if action == "1":
        lang = input("Целевой язык [ru]: ").strip() or "ru"
        translator = input("Переводчик (google/llama/openrouter/sprut) [google]: ").strip() or "google"
        output_html = f"{input_base}_translate.html"
        print(f"\nЗапуск перевода -> {output_html}\n")
        result = process_pdf(
            pdf_path=selected_pdf, output_html=output_html, lang=lang,
            translator_type=translator, quiet=False,
        )
    elif action == "2":
        output_html = f"{input_base}_summary.html"
        print(f"\nЗапуск генерации реферата -> {output_html}\n")
        result = process_pdf(
            pdf_path=selected_pdf, output_html=output_html, summary_mode=True, quiet=False,
        )
    else:
        output_html = f"{input_base}_raw.html"
        print(f"\nСохранение в HTML -> {output_html}\n")
        result = process_pdf_raw(pdf_path=selected_pdf, output_html=output_html)

    stats = result.get("stats", {})
    timings = stats.get("timings", {})
    logger.info("=" * 60)
    if timings:
        logger.info("  ПРОФИЛИРОВАНИЕ:")
        for stage, t in timings.items():
            logger.info(f"   {stage}: {_format_time(t)}")
    logger.info("=" * 60)
    logger.info(f" Готово: {output_html}")


def main():
    if len(sys.argv) == 1:
        interactive_mode()
        return

    parser = argparse.ArgumentParser(description="PDF Translator — единый скрипт")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input", help="Входной PDF файл")
    parser.add_argument("-l", "--lang", default="ru", help="Целевой язык (по умолчанию: ru)")
    parser.add_argument("-t", "--translator", default="google", choices=["google", "llama", "openrouter", "sprut"],
                        help="Переводчик (по умолчанию: google)")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1", help="URL llama.cpp сервера")
    parser.add_argument("--llama-model", help="Ожидаемое имя модели")
    parser.add_argument("--sprut-model", default=os.getenv("SPRUT_MODEL"),
                        help="Модель для SprutDock (например, z-ai/glm-5.2:free)")
    parser.add_argument("--no-auto-find", action="store_true", help="Отключить автоматический поиск сервера")
    parser.add_argument("--summary", action="store_true", help="Режим реферата")
    parser.add_argument("--raw-html", action="store_true", help="Сохранение в HTML без перевода")
    parser.add_argument("--summary-translator", choices=["google", "llama", "openrouter", "sprut"],
                        default="llama", help="Генератор реферата (по умолчанию: llama)")
    parser.add_argument("--summary-lang-translator", choices=["google", "llama", "openrouter", "sprut"],
                        default="google", help="Переводчик реферата (по умолчанию: google)")
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 1) * 2), help="Число воркеров")
    parser.add_argument("--block-timeout", type=int, default=BLOCK_TIMEOUT, help="Максимум секунд на один блок")
    parser.add_argument("--dark-html", action="store_true", default=True, help="Тёмная тема HTML (по умолчанию: вкл)")
    parser.add_argument("--light-html", action="store_true", help="Светлая тема HTML")
    parser.add_argument("-q", "--quiet", action="store_true", help="Тихий режим")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)

    dark_html = not args.light_html
    input_base = os.path.splitext(args.input)[0]

    if args.raw_html:
        output_html = f"{input_base}_raw.html"
    elif args.summary:
        output_html = f"{input_base}_summary.html"
    else:
        output_html = f"{input_base}_translate.html"

    logger.info("=" * 60)
    if args.raw_html:
        logger.info(f"PDF TRANSLATOR — RAW HTML v{__version__}")
    else:
        logger.info(f"PDF TRANSLATOR — {args.translator.upper()} v{__version__}")
    logger.info("=" * 60)

    try:
        if args.raw_html:
            result = process_pdf_raw(
                pdf_path=args.input, output_html=output_html, dark_html=dark_html,
            )
        else:
            result = process_pdf(
                pdf_path=args.input, output_html=output_html, lang=args.lang, translator_type=args.translator,
                llama_url=args.llama_url, llama_model=args.llama_model, auto_find=not args.no_auto_find,
                max_workers=args.workers, timeout=args.block_timeout, quiet=args.quiet, summary_mode=args.summary,
                summary_translator_type=args.summary_translator, translate_translator_type=args.summary_lang_translator,
                dark_html=dark_html, sprut_model=args.sprut_model,
            )

        stats = result["stats"]
        logger.info("=" * 60)
        timings = stats.get("timings", {})
        if timings:
            logger.info("  ПРОФИЛИРОВАНИЕ:")
            for stage, t in timings.items():
                logger.info(f"   {stage}: {_format_time(t)}")
        elif stats.get("summary_mode", False):
            logger.info(" СТАТИСТИКА ГЕНЕРАЦИИ РЕФЕРАТА:")
            logger.info(f"    Обработано чанков: {stats.get('chunks', 0)}")
            if stats.get("tokens", 0) > 0:
                logger.info(f"    Сгенерировано токенов (приблиз.): {stats['tokens']}")
            if stats.get("speed", 0) > 0:
                logger.info(f"    Скорость: {stats['speed']:.1f} токенов/сек")
            if stats.get("model"):
                logger.info(f"    Модель: {stats['model']}")
            if stats.get("time", 0) > 0:
                logger.info(f"    Время генерации: {_format_time(stats['time'])}")
        else:
            logger.info(" СТАТИСТИКА ПЕРЕВОДА:")
            logger.info(f"    Переведено: {stats.get('success', 0)}")
            logger.info(f"    Из кэша:   {stats.get('cached', 0)}")
            logger.info(f"    Пропущено:  {stats.get('skipped', 0)}")
            logger.info(f"    Ошибок:     {stats.get('failed', 0)}")
            if stats.get("error"):
                logger.error(f"   Причина: {stats['error']}")
        logger.info("=" * 60)
        logger.info(f" Готово: {output_html}")

    except KeyboardInterrupt:
        logger.warning(" Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f" Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
