"""
Database storage using aiosqlite for persistence.
Supports message history, per-user dynamic settings, token statistics, and deduplication.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiosqlite


class Database:
    def __init__(self, db_path: str = "data/assistant.db") -> None:
        self.db_path = db_path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Initialize database schema with required tables and indexes."""
        self._ensure_dir()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages (user_id, id);
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    model TEXT,
                    thinking_level TEXT,
                    context_messages INTEGER,
                    search_enabled INTEGER,
                    max_output_tokens INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            try:
                await db.execute("ALTER TABLE user_settings ADD COLUMN max_output_tokens INTEGER;")
            except Exception:
                pass

            await db.execute("""
                CREATE TABLE IF NOT EXISTS usage_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    candidate_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    search_queries INTEGER DEFAULT 0,
                    model_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage_stats (user_id);
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS session_memory (
                    user_id TEXT PRIMARY KEY,
                    summary TEXT DEFAULT '',
                    compressed_msg_id INTEGER DEFAULT 0,
                    last_active_at REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await db.commit()

    async def save_message(self, user_id: str, role: str, content: str) -> None:
        """Save a single message to history."""
        # Sanitize surrogates to ensure strict UTF-8 compliance in SQLite
        clean_content = content.encode("utf-8", errors="replace").decode("utf-8")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?);",
                (str(user_id), role, clean_content)
            )
            await db.commit()

    async def get_recent_messages(self, user_id: str, limit: Optional[int]) -> List[Dict[str, str]]:
        """
        Get the most recent N messages for a user, returned in chronological order.
        """
        if limit is None or limit <= 0:
            return []
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, id FROM messages
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC;
                """,
                (str(user_id), limit)
            )
            rows = await cursor.fetchall()
            return [{"role": row[0], "content": row[1]} for row in rows]

    async def clear_history(self, user_id: str) -> int:
        """Clear all conversation history and session summary for a specific user."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM messages WHERE user_id = ?;",
                (str(user_id),)
            )
            await db.execute(
                "DELETE FROM session_memory WHERE user_id = ?;",
                (str(user_id),)
            )
            await db.commit()
            return cursor.rowcount

    async def get_session_memory(self, user_id: str) -> Dict[str, Any]:
        """Get the active session memory and summary state for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT summary, compressed_msg_id, last_active_at FROM session_memory WHERE user_id = ?;",
                (str(user_id),)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "summary": row[0] or "",
                    "compressed_msg_id": row[1] or 0,
                    "last_active_at": float(row[2]) if row[2] else 0.0,
                }
            return {
                "summary": "",
                "compressed_msg_id": 0,
                "last_active_at": 0.0,
            }

    async def update_session_memory(
        self,
        user_id: str,
        summary: Optional[str] = None,
        compressed_msg_id: Optional[int] = None,
        last_active_at: Optional[float] = None,
    ) -> None:
        """Upsert user's session memory state."""
        existing = await self.get_session_memory(user_id)
        new_summary = summary if summary is not None else existing.get("summary", "")
        new_msg_id = compressed_msg_id if compressed_msg_id is not None else existing.get("compressed_msg_id", 0)
        new_active = last_active_at if last_active_at is not None else existing.get("last_active_at", 0.0)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO session_memory (user_id, summary, compressed_msg_id, last_active_at, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    summary = excluded.summary,
                    compressed_msg_id = excluded.compressed_msg_id,
                    last_active_at = excluded.last_active_at,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (str(user_id), new_summary, new_msg_id, new_active)
            )
            await db.commit()

    async def clear_session_memory(self, user_id: str) -> None:
        """Clear session memory and summary for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE session_memory SET summary = '', compressed_msg_id = 0, last_active_at = 0, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?;",
                (str(user_id),)
            )
            await db.commit()

    async def get_uncompressed_messages(self, user_id: str, compressed_msg_id: int = 0) -> List[Dict[str, Any]]:
        """Get all messages for a user with id > compressed_msg_id in chronological order."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT id, role, content FROM messages
                WHERE user_id = ? AND id > ?
                ORDER BY id ASC;
                """,
                (str(user_id), compressed_msg_id)
            )
            rows = await cursor.fetchall()
            return [{"id": row[0], "role": row[1], "content": row[2]} for row in rows]

    async def get_user_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get custom user settings if configured."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT model, thinking_level, context_messages, search_enabled, max_output_tokens FROM user_settings WHERE user_id = ?;",
                (str(user_id),)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "model": row[0],
                    "thinking_level": row[1],
                    "context_messages": row[2],
                    "search_enabled": bool(row[3]) if row[3] is not None else None,
                    "max_output_tokens": row[4],
                }
            return None

    async def update_user_settings(
        self,
        user_id: str,
        model: Optional[str] = None,
        thinking_level: Optional[str] = None,
        context_messages: Optional[int] = None,
        search_enabled: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        """Upsert user customized settings."""
        existing = await self.get_user_settings(user_id) or {}
        new_model = model if model is not None else existing.get("model")
        new_level = thinking_level if thinking_level is not None else existing.get("thinking_level")
        new_ctx = context_messages if context_messages is not None else existing.get("context_messages")
        new_search = (
            int(search_enabled)
            if search_enabled is not None
            else (int(existing["search_enabled"]) if "search_enabled" in existing and existing["search_enabled"] is not None else None)
        )
        new_max_tokens = max_output_tokens if max_output_tokens is not None else existing.get("max_output_tokens")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO user_settings (user_id, model, thinking_level, context_messages, search_enabled, max_output_tokens, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    model = excluded.model,
                    thinking_level = excluded.thinking_level,
                    context_messages = excluded.context_messages,
                    search_enabled = excluded.search_enabled,
                    max_output_tokens = excluded.max_output_tokens,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (str(user_id), new_model, new_level, new_ctx, new_search, new_max_tokens)
            )
            await db.commit()

    async def record_usage(
        self,
        user_id: str,
        prompt_tokens: int,
        candidate_tokens: int,
        search_queries: int = 0,
        model_used: str = "",
    ) -> None:
        """Record token usage and search query statistics for cost estimation."""
        total = prompt_tokens + candidate_tokens
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO usage_stats (user_id, prompt_tokens, candidate_tokens, total_tokens, search_queries, model_used)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (str(user_id), prompt_tokens, candidate_tokens, total, search_queries, model_used)
            )
            await db.commit()

    async def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Aggregate total usage for a user and across all users."""
        async with aiosqlite.connect(self.db_path) as db:
            # User stats
            cursor = await db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(candidate_tokens), 0),
                       COALESCE(SUM(total_tokens), 0), COALESCE(SUM(search_queries), 0)
                FROM usage_stats WHERE user_id = ?;
                """,
                (str(user_id),)
            )
            u_row = await cursor.fetchone()

            # Global stats
            cursor_global = await db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(candidate_tokens), 0),
                       COALESCE(SUM(total_tokens), 0), COALESCE(SUM(search_queries), 0)
                FROM usage_stats;
                """
            )
            g_row = await cursor_global.fetchone()

            return {
                "user": {
                    "requests": u_row[0],
                    "prompt_tokens": u_row[1],
                    "candidate_tokens": u_row[2],
                    "total_tokens": u_row[3],
                    "search_queries": u_row[4],
                },
                "global": {
                    "requests": g_row[0],
                    "prompt_tokens": g_row[1],
                    "candidate_tokens": g_row[2],
                    "total_tokens": g_row[3],
                    "search_queries": g_row[4],
                }
            }

    async def is_message_processed(self, message_id: str) -> bool:
        """Check if message_id has already been processed."""
        if not message_id:
            return False
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM processed_messages WHERE message_id = ?;",
                (str(message_id),)
            )
            return (await cursor.fetchone()) is not None

    async def mark_message_processed(self, message_id: str) -> None:
        """Mark message_id as processed."""
        if not message_id:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO processed_messages (message_id) VALUES (?);",
                (str(message_id),)
            )
            await db.commit()
