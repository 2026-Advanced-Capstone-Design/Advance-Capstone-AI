import os
import sqlite3

_DB_PATH = os.getenv("CACHE_DB_PATH", "./cache.db")


def _conn():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_cache (
            topic_key   TEXT PRIMARY KEY,
            background  TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def get_background(topic_key: str) -> str | None:
    """캐시에서 배경 지식을 조회합니다."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT background FROM knowledge_cache WHERE topic_key = ?",
            (topic_key,),
        ).fetchone()
    return row[0] if row else None


def set_background(topic_key: str, background: str) -> None:
    """배경 지식을 캐시에 저장합니다."""
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO knowledge_cache (topic_key, background) VALUES (?, ?)",
            (topic_key, background),
        )
        conn.commit()
