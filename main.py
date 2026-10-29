"""
Main application entry point for the QQ Gemini Assistant.
Supports both production OneBot 11 service and interactive console testing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path

from config import get_settings
from db import Database
from gemini_client import GoogleGenAIClient, BaseGeminiClient
from mock_gemini import MockGeminiClient
from chat_manager import ChatManager
from qq_adapter.onebot_v11 import OneBotV11Adapter
from qq_adapter.console_adapter import ConsoleAdapter


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="QQ Gemini Assistant")
    parser.add_argument("--console", action="store_true", help="Run interactive console simulation")
    parser.add_argument("--mock", action="store_true", help="Force use of MockGeminiClient")
    parser.add_argument("--whitelist", type=str, help="Override QQ whitelist (comma-separated)")
    args = parser.parse_args()

    settings = get_settings()
    if args.whitelist:
        settings.qq_whitelist_raw = args.whitelist

    setup_logging(settings.log_level)
    logger = logging.getLogger("main")

    logger.info("Starting QQ Gemini Assistant...")
    logger.info(f"Target Model: {settings.gemini_model}")
    logger.info(f"Allowed QQ Whitelist: {list(settings.qq_whitelist)}")
    logger.info(f"Default Thinking Level: {settings.default_thinking_level}")
    logger.info(f"Search Grounding Default: {settings.enable_search_default}")

    # Initialize SQLite Database
    db = Database(settings.sqlite_db_path)
    await db.init_db()
    logger.info(f"Database initialized at {settings.sqlite_db_path}")

    # Determine Gemini Client
    client: BaseGeminiClient
    if args.mock or not settings.gemini_api_key:
        if not args.mock and not settings.gemini_api_key:
            logger.warning("No GEMINI_API_KEY found in environment. Automatically falling back to MockGeminiClient.")
        else:
            logger.info("Using MockGeminiClient as requested.")
        client = MockGeminiClient(default_model=settings.gemini_model)
    else:
        logger.info("Initializing official GoogleGenAIClient...")
        client = GoogleGenAIClient(
            api_key=settings.gemini_api_key,
            default_model=settings.gemini_model,
            default_thinking_level=settings.default_thinking_level,
            default_max_output_tokens=settings.max_output_tokens,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_retries=settings.gemini_max_retries,
        )

    # Initialize ChatManager
    chat_manager = ChatManager(settings=settings, db=db, client=client)

    if args.console:
        # Run in interactive console mode
        console = ConsoleAdapter(settings=settings, chat_manager=chat_manager)
        await console.run_interactive_loop()
    else:
        # Run OneBot 11 Adapter
        adapter = OneBotV11Adapter(settings=settings, chat_manager=chat_manager)
        await adapter.start()

        # Keep running until SIGINT or SIGTERM
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received.")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Windows signal handling fallback
                pass

        try:
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await adapter.stop()
            logger.info("Assistant shut down cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess terminated by user.")
        sys.exit(0)
