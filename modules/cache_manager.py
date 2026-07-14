# cache_manager.py
import sqlite3
import hashlib
import time
from pathlib import Path

class TranslationCache:
    def __init__(self, db_path: str = "translation_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    text_hash TEXT,
                    lang TEXT,
                    translation TEXT,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (text_hash, lang)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created)")
            # Очистка старого кэша (старше 30 дней)
            conn.execute("DELETE FROM cache WHERE created < datetime('now', '-30 days')")
            conn.commit()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str, lang: str) -> str | None:
        h = self._hash(text)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT translation FROM cache WHERE text_hash=? AND lang=?", (h, lang))
            row = cur.fetchone()
            return row[0] if row else None

    def put(self, text: str, lang: str, translation: str):
        h = self._hash(text)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (text_hash, lang, translation) VALUES (?,?,?)",
                (h, lang, translation)
            )
            conn.commit()