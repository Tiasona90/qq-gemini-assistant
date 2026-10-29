"""
Console / CLI Adapter for interactive simulation and manual testing without QQ.
"""

from __future__ import annotations

import asyncio
import sys
import time

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from chat_manager import ChatManager
from config import Settings
from image_utils import extract_message_and_images


class ConsoleAdapter:
    def __init__(self, settings: Settings, chat_manager: ChatManager) -> None:
        self.settings = settings
        self.chat_manager = chat_manager
        # Use first whitelisted QQ ID or default test ID
        whitelist = list(self.settings.qq_whitelist)
        self.default_user_id = whitelist[0] if whitelist else "10001"

    async def run_interactive_loop(self) -> None:
        print("=" * 60)
        print("  QQ Gemini 智能助手 - 控制台模拟交互终端")
        print(f"  当前模拟发送者 QQ 号: {self.default_user_id}")
        print("  输入消息后回车发送。支持 /help, /reset, //成本, //低思考 等指令。")
        print("  输入 'exit' 或 'quit' 退出交互。")
        print("=" * 60)

        loop = asyncio.get_event_loop()
        msg_seq = 1

        while True:
            try:
                line = await loop.run_in_executor(None, input, f"\n[QQ用户 {self.default_user_id}] > ")
            except (EOFError, KeyboardInterrupt):
                break

            text = line.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                print("退出控制台模拟模式。")
                break

            msg_id = f"console_msg_{int(time.time())}_{msg_seq}"
            msg_seq += 1

            clean_text, img_sources = extract_message_and_images({"raw_message": text})

            reply = await self.chat_manager.process_message(
                user_id=self.default_user_id,
                content=clean_text,
                message_id=msg_id,
                image_sources=img_sources,
            )

            print("\n[QQ助手 回复]")
            print(reply if reply else "[无回复/已静默处理]")
            print("-" * 50)
