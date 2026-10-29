"""
Mock Gemini Client for offline testing, local simulation, and edge cases.
Implements the BaseGeminiClient protocol and allows configurable mock behaviors.
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Dict, Any, Optional
from gemini_client import BaseGeminiClient, GeminiResponse, GroundingSource


class MockGeminiClient(BaseGeminiClient):
    def __init__(
        self,
        default_model: str = "mock-gemini-3.8-flash",
        simulate_delay: float = 0.05,
    ) -> None:
        self.default_model = default_model
        self.simulate_delay = simulate_delay
        self.call_count = 0
        self.last_messages: List[Dict[str, str]] = []
        self.last_images: List[Dict[str, Any]] = []
        self.last_enable_search: bool = False
        self.last_thinking_level: Optional[str] = None

        # Fault injection controls
        self.fail_with_timeout_count: int = 0
        self.fail_with_429_count: int = 0
        self.fail_with_auth_error: bool = False

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        thinking_level: Optional[str] = None,
        enable_search: bool = False,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> GeminiResponse:
        self.call_count += 1
        self.last_messages = list(messages)
        self.last_images = list(images) if images else []
        self.last_enable_search = enable_search
        self.last_thinking_level = thinking_level
        target_model = model or self.default_model

        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)

        # Fault injection: Authentication error
        if self.fail_with_auth_error:
            from google.genai.errors import APIError
            err = APIError(403, {"error": {"code": 403, "message": "API_KEY_INVALID: API key not valid."}})
            raise err

        # Fault injection: Timeout
        if self.fail_with_timeout_count > 0:
            self.fail_with_timeout_count -= 1
            raise asyncio.TimeoutError("Simulated Gemini API timeout")

        # Fault injection: 429 Rate Limit
        if self.fail_with_429_count > 0:
            self.fail_with_429_count -= 1
            from google.genai.errors import APIError
            err = APIError(429, {"error": {"code": 429, "message": "ResourceExhausted: Quota exceeded."}})
            raise err

        # Normal generation
        user_msg = messages[-1]["content"] if messages else ""

        # Check if this is a session memory compression request
        is_summary = (
            (system_instruction and ("对话记忆提炼" in system_instruction or "背景摘要" in system_instruction))
            or ("合并更新后的背景摘要" in user_msg)
        )
        if is_summary:
            return GeminiResponse(
                text="【前文背景摘要】用户与助手进行了多轮深度探讨，梳理了关键需求与核心参数，明确了主要事实与后续方案。",
                prompt_tokens=200,
                candidate_tokens=60,
                total_tokens=260,
                model_used=target_model,
            )

        # Check if this is a fuzzy command parse request
        is_extract = (
            (system_instruction and ("参数提取" in system_instruction or "配置" in system_instruction))
            or ("JSON" in user_msg and "//" in user_msg)
        )
        if is_extract:
            import re
            cmd_match = re.search(r"用户指令:\s*//(.*?)(?:\n|$)", user_msg)
            actual_cmd = cmd_match.group(1) if cmd_match else user_msg

            # Extract current state from prompt if available
            cur_model_match = re.search(r"当前模型:\s*([a-zA-Z0-9\.\-_]+)", user_msg)
            cur_model = cur_model_match.group(1).lower() if cur_model_match else "gemini-3.8-flash"

            cur_think_match = re.search(r"当前思考强度:\s*([a-zA-Z]+)", user_msg)
            cur_think = cur_think_match.group(1).upper() if cur_think_match else "LOW"

            mock_ctx = None
            if "10轮" in actual_cmd or "十轮" in actual_cmd:
                mock_ctx = 20
            elif "5轮" in actual_cmd or "五轮" in actual_cmd:
                mock_ctx = 10
            elif "8条" in actual_cmd or "八条" in actual_cmd:
                mock_ctx = 8

            mock_think = None
            is_thinking_cmd = any(w in actual_cmd for w in ["思考", "深度", "想一想"])
            if is_thinking_cmd:
                ladder = ["LOW", "MEDIUM", "HIGH"]
                if "更高" in actual_cmd or "高一点" in actual_cmd or "更强" in actual_cmd:
                    idx = ladder.index(cur_think) if cur_think in ladder else 0
                    mock_think = ladder[min(idx + 1, len(ladder) - 1)]
                elif "更低" in actual_cmd or "低一点" in actual_cmd or "更弱" in actual_cmd:
                    idx = ladder.index(cur_think) if cur_think in ladder else 0
                    mock_think = ladder[max(idx - 1, 0)]
                elif "高" in actual_cmd:
                    mock_think = "HIGH"
                elif "中" in actual_cmd:
                    mock_think = "MEDIUM"
                elif any(w in actual_cmd for w in ["低", "最小", "极低", "关闭", "无"]):
                    mock_think = "LOW"

            mock_search = None
            if "开启搜索" in actual_cmd or "打开搜索" in actual_cmd:
                mock_search = True
            elif "关闭搜索" in actual_cmd:
                mock_search = False

            mock_model = None
            if any(w in actual_cmd for w in ["更好", "更强", "升级", "聪明", "高级", "顶配", "好一点"]):
                if "3.5" in cur_model:
                    mock_model = "gemini-3.8-flash"
                else:
                    mock_model = "gemini-4-argon"
            elif any(w in actual_cmd for w in ["便宜", "省钱", "轻量", "低级", "基础", "降级", "差一点", "低配", "低一点"]):
                if "4" in cur_model:
                    mock_model = "gemini-3.8-flash"
                else:
                    mock_model = "gemini-3.5-flash-lite"
            elif "argon" in actual_cmd or "4" in actual_cmd:
                mock_model = "gemini-4-argon"
            elif "3.8" in actual_cmd:
                mock_model = "gemini-3.8-flash"
            elif "3.5" in actual_cmd:
                mock_model = "gemini-3.5-flash-lite"

            resp_dict = {
                "thinking_level": mock_think,
                "context_messages": mock_ctx,
                "search_enabled": mock_search,
                "model": mock_model,
                "max_output_tokens": None,
            }
            # Wrap in slight conversational text or markdown to test parser extraction robustness
            resp_text = f"```json\n{json.dumps(resp_dict, ensure_ascii=False)}\n```"
            return GeminiResponse(
                text=resp_text,
                prompt_tokens=50,
                candidate_tokens=25,
                total_tokens=75,
                model_used=target_model,
            )

        sources: List[GroundingSource] = []
        queries: List[str] = []

        if enable_search:
            queries = [f"实时搜索: {user_msg[:30]}"]
            sources = [
                GroundingSource(
                    title="新华网实时资讯",
                    url="https://www.news.cn/latest/2026-10-sample"
                ),
                GroundingSource(
                    title="百度百科参考条目",
                    url="https://baike.baidu.com/item/sample"
                )
            ]
            resp_text = f"这是模拟的联网检索回复，针对问题：{user_msg}"
        else:
            if self.last_images:
                resp_text = f"已识别你发送的 {len(self.last_images)} 张图片。针对问题“{user_msg}”：这是对图片的详细解析。"
            else:
                history_len = len(messages) - 1
                if history_len > 0:
                    resp_text = f"收到你的多轮提问（已携带此前 {history_len} 条历史消息）：{user_msg}"
                else:
                    resp_text = f"你好，我是 Gemini 智能助手。关于你的问题：{user_msg}，以下是简洁解答。"

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        p_tok = max(10, prompt_chars // 2)
        c_tok = max(15, len(resp_text) // 2)

        return GeminiResponse(
            text=resp_text,
            prompt_tokens=p_tok,
            candidate_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            search_queries=queries,
            sources=sources,
            model_used=target_model,
        )
