#!/usr/bin/env python3
"""Unit-тесты для main2.py — классификация, кэш, утилиты, переводчики."""

import os
import sys
import json
import time
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

# Мокаем fitz если не установлен
sys.modules.setdefault('fitz', MagicMock())
sys.modules.setdefault('pymupdf_layout', MagicMock())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main2 import (
    _format_time,
    _chunk_text_by_sentences,
    _count_list_lines,
    _extract_features,
    make_span,
    make_line,
    make_block,
    block_metrics,
    BlockClassifier,
    _classifier,
    REF_HEADING_RE,
    LIST_LINE_RE,
    CAPTION_RE,
    RateLimiter,
    TranslationCache,
    GoogleTranslator,
    stage_timer,
    intersection_ratio,
)


class TestFormatTime(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(_format_time(5), "5с")

    def test_minutes(self):
        self.assertEqual(_format_time(90), "1м 30с")

    def test_hours(self):
        self.assertEqual(_format_time(3661), "1ч 01м")


class TestChunkTextBySentences(unittest.TestCase):
    def test_short_text(self):
        result = _chunk_text_by_sentences("Hello world.")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Hello world.")

    def test_long_text(self):
        text = ". ".join(f"Sentence {i}." for i in range(200))
        chunks = _chunk_text_by_sentences(text, max_chunk_size=500)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 700)

    def test_empty(self):
        result = _chunk_text_by_sentences("")
        self.assertEqual(len(result), 1)


class TestListDetection(unittest.TestCase):
    def test_bullet_list(self):
        text = "• Item one\n• Item two\n• Item three"
        self.assertEqual(_count_list_lines(text), 3)

    def test_numbered_list(self):
        text = "1. First\n2. Second\n3. Third"
        self.assertEqual(_count_list_lines(text), 3)

    def test_mixed_list(self):
        text = "- Dash item\n* Star item"
        self.assertEqual(_count_list_lines(text), 2)

    def test_no_list(self):
        text = "Just a normal paragraph with no lists."
        self.assertEqual(_count_list_lines(text), 0)

    def test_regex_patterns(self):
        self.assertTrue(LIST_LINE_RE.match("• bullet"))
        self.assertTrue(LIST_LINE_RE.match("- dash"))
        self.assertTrue(LIST_LINE_RE.match("* star"))
        self.assertTrue(LIST_LINE_RE.match("1. numbered"))
        self.assertTrue(LIST_LINE_RE.match("a. alpha"))
        self.assertIsNone(LIST_LINE_RE.match("No marker here"))


class TestRegexPatterns(unittest.TestCase):
    def test_ref_heading(self):
        self.assertTrue(REF_HEADING_RE.match("References"))
        self.assertTrue(REF_HEADING_RE.match("Bibliography"))
        self.assertTrue(REF_HEADING_RE.match("Литература"))
        self.assertTrue(REF_HEADING_RE.match("Список литературы"))
        self.assertIsNone(REF_HEADING_RE.match("Some other heading"))

    def test_caption_re(self):
        self.assertTrue(CAPTION_RE.search("Figure 1: Results"))
        self.assertTrue(CAPTION_RE.search("Table 2: Data"))
        self.assertTrue(CAPTION_RE.search("Рис. 3"))
        self.assertIsNone(CAPTION_RE.search("Just text"))


class TestBlockMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        span = make_span(text="Hello", size=14, flags=1)
        line = make_line(spans=[span])
        block = make_block(type="paragraph", lines=[line], bbox=(0, 0, 100, 50))
        m = block_metrics(block, page_width=612)
        self.assertEqual(m.text, "Hello")
        self.assertEqual(m.max_font, 14)
        self.assertFalse(m.all_upper)
        self.assertTrue(m.has_bold)

    def test_empty_block(self):
        block = make_block(type="empty", lines=[])
        m = block_metrics(block, page_width=612)
        self.assertEqual(m.text, "")
        self.assertEqual(m.total_spans, 0)


class TestBlockClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = BlockClassifier()
        self.page_width = 612.0
        self.avg_font = 12.0

    def test_empty_block(self):
        block = make_block(type="text", lines=[])
        self.assertEqual(self.classifier.classify(block, self.page_width, self.avg_font), "empty")

    def test_reference_heading(self):
        span = make_span(text="References", size=14, flags=1)
        line = make_line(spans=[span])
        block = make_block(type="text", lines=[line], bbox=(0, 0, 612, 30))
        self.assertEqual(self.classifier.classify(block, self.page_width, self.avg_font), "reference_heading")

    def test_bibliography(self):
        span = make_span(text="Библиография", size=14, flags=1)
        line = make_line(spans=[span])
        block = make_block(type="text", lines=[line], bbox=(0, 0, 612, 30))
        self.assertEqual(self.classifier.classify(block, self.page_width, self.avg_font), "reference_heading")

    def test_numbered_reference(self):
        span = make_span(text="[1] Author, Title, 2020.", size=10, flags=0)
        line = make_line(spans=[span])
        block = make_block(type="text", lines=[line], bbox=(0, 0, 400, 20))
        self.assertEqual(self.classifier.classify(block, self.page_width, self.avg_font), "reference")

    def test_list_block(self):
        text = "• Item one\n• Item two\n• Item three"
        span = make_span(text=text, size=12, flags=0)
        line = make_line(spans=[span])
        block = make_block(type="text", lines=[line], bbox=(0, 0, 400, 60))
        result = self.classifier.classify(block, self.page_width, self.avg_font)
        self.assertEqual(result, "list")

    def test_metadata_doi(self):
        span = make_span(text="DOI: 10.1234/example", size=10, flags=0)
        line = make_line(spans=[span])
        block = make_block(type="text", lines=[line], bbox=(0, 0, 400, 20))
        self.assertEqual(self.classifier.classify(block, self.page_width, self.avg_font), "metadata")

    def test_paragraph(self):
        span = make_span(text="This is a regular paragraph with some text.", size=12, flags=0)
        line = make_line(spans=[span])
        block = make_block(type="text", lines=[line], bbox=(50, 100, 550, 130))
        result = self.classifier.classify(block, self.page_width, self.avg_font)
        self.assertIn(result, ("paragraph", "heading"))

    def test_heading_large_font(self):
        span = make_span(text="CHAPTER 1", size=20, flags=1)
        line = make_line(spans=[span])
        block = make_block(type="text", lines=[line], bbox=(200, 50, 400, 80))
        result = self.classifier.classify(block, self.page_width, self.avg_font)
        self.assertEqual(result, "heading")


class TestIntersectionRatio(unittest.TestCase):
    def test_no_intersection(self):
        r = intersection_ratio((0, 0, 10, 10), (20, 20, 30, 30))
        self.assertEqual(r, 0.0)

    def test_full_intersection(self):
        r = intersection_ratio((0, 0, 10, 10), (0, 0, 10, 10))
        self.assertAlmostEqual(r, 1.0)

    def test_partial_intersection(self):
        r = intersection_ratio((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertAlmostEqual(r, 0.25)

    def test_zero_area_block(self):
        r = intersection_ratio((5, 5, 5, 5), (0, 0, 10, 10))
        self.assertEqual(r, 0.0)


class TestRateLimiter(unittest.TestCase):
    def test_basic(self):
        rl = RateLimiter(max_requests_per_second=10)
        t0 = time.monotonic()
        rl.acquire()
        rl.acquire()
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.05)


class TestTranslationCache(unittest.TestCase):
    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        )
        self._tmpfile.close()
        self.cache_path = self._tmpfile.name
        TranslationCache._instance = None

    def tearDown(self):
        TranslationCache._instance = None
        if os.path.exists(self.cache_path):
            os.unlink(self.cache_path)

    def test_put_get(self):
        cache = TranslationCache(self.cache_path)
        cache.put("hello", "ru", "привет")
        self.assertEqual(cache.get("hello", "ru"), "привет")

    def test_miss(self):
        cache = TranslationCache(self.cache_path)
        self.assertIsNone(cache.get("nonexistent", "ru"))

    def test_save_load(self):
        cache = TranslationCache(self.cache_path)
        cache.put("test", "en", "result")
        cache.save()
        TranslationCache._instance = None
        cache2 = TranslationCache(self.cache_path)
        self.assertEqual(cache2.get("test", "en"), "result")

    def test_key_uniqueness(self):
        cache = TranslationCache(self.cache_path)
        cache.put("hello", "ru", "привет")
        cache.put("hello", "en", "hello")
        self.assertEqual(cache.get("hello", "ru"), "привет")
        self.assertEqual(cache.get("hello", "en"), "hello")


class TestExtractFeatures(unittest.TestCase):
    def test_features_length(self):
        span = make_span(text="Test text", size=12, flags=0)
        line = make_line(spans=[span])
        block = make_block(type="paragraph", lines=[line], bbox=(0, 0, 100, 50))
        features = _extract_features(block, 612.0, 12.0)
        self.assertEqual(len(features), 15)


class TestStageTimer(unittest.TestCase):
    def test_stage_timer(self):
        timings = {}
        with stage_timer("test_stage", timings):
            time.sleep(0.01)
        self.assertIn("test_stage", timings)
        self.assertGreater(timings["test_stage"], 0)


class TestGoogleTranslator(unittest.TestCase):
    def test_generate_returns_none(self):
        tr = GoogleTranslator.__new__(GoogleTranslator)
        tr.name = "Google"
        self.assertIsNone(tr.generate("test prompt"))

    def test_translate_error_handling(self):
        tr = GoogleTranslator.__new__(GoogleTranslator)
        tr.name = "Google"
        tr.target_lang = "ru"
        tr.session = MagicMock()
        tr.session.get.side_effect = Exception("connection error")
        tr.rate_limiter = MagicMock()
        result = tr.translate("test")
        self.assertIsNone(result)


class TestMakeBlockFactories(unittest.TestCase):
    def test_make_span(self):
        s = make_span(text="Hello", size=14)
        self.assertEqual(s.text, "Hello")
        self.assertEqual(s.size, 14)

    def test_make_line(self):
        span = make_span(text="test")
        line = make_line(spans=[span])
        self.assertEqual(len(line.spans), 1)

    def test_make_block(self):
        block = make_block(type="heading", page_num=1)
        self.assertEqual(block.type, "heading")
        self.assertEqual(block.page_num, 1)

    def test_block_text_property(self):
        span1 = make_span(text="Hello")
        span2 = make_span(text="World")
        line = make_line(spans=[span1, span2])
        block = make_block(type="paragraph", lines=[line])
        self.assertEqual(block.text, "Hello World")


if __name__ == "__main__":
    unittest.main()
