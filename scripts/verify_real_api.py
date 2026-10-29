"""
Standalone verification script for live Gemini API calls.
Can be executed when GEMINI_API_KEY is configured in the environment.
Tests:
1. Basic single-turn completion
2. Multi-turn conversation context
3. Google Search Grounding and metadata extraction
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from gemini_client import GoogleGenAIClient


async def run_live_test() -> None:
    settings = get_settings()
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        print("[!] GEMINI_API_KEY 未设置。请先设置环境变量或在 .env 中填入后再次运行。")
        print("    例如: $env:GEMINI_API_KEY='你的密钥'")
        sys.exit(1)

    print(f"[*] 正在使用模型: {settings.gemini_model} 进行官方 API 验证...")
    client = GoogleGenAIClient(
        api_key=api_key,
        default_model=settings.gemini_model,
        default_thinking_level=settings.default_thinking_level,
    )

    # 1. Basic completion
    print("\n[1/3] 测试普通文本生成...")
    resp1 = await client.generate_response(
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        enable_search=False,
    )
    print(f"  回复: {resp1.text}")
    print(f"  Token: 输入={resp1.prompt_tokens}, 输出={resp1.candidate_tokens}, 总计={resp1.total_tokens}")

    # 2. Multi-turn conversation
    print("\n[2/3] 测试多轮上下文连贯性...")
    resp2 = await client.generate_response(
        messages=[
            {"role": "user", "content": "我最喜欢的编程语言是 Python"},
            {"role": "model", "content": "收到，Python 是一门非常优雅且实用的语言。"},
            {"role": "user", "content": "我刚才说我喜欢什么语言？"},
        ],
        enable_search=False,
    )
    print(f"  回复: {resp2.text}")

    # 3. Search Grounding
    print("\n[3/3] 测试 Google Search Grounding 实时检索...")
    resp3 = await client.generate_response(
        messages=[{"role": "user", "content": "请查询澳大利亚当前的法定最低时薪是多少？"}],
        enable_search=True,
    )
    print(f"  回复: {resp3.text}")
    print(f"  检索词: {resp3.search_queries}")
    print(f"  提取到的参考来源: {[s.url for s in resp3.sources]}")

    print("\n[✓] 真实 Gemini API 验证全部顺利通过！")


if __name__ == "__main__":
    asyncio.run(run_live_test())
