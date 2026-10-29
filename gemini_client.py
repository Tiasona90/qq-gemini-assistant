"""
Gemini Client implementation using the official google-genai SDK.
Handles content generation, thinking levels, Google Search Grounding,
exponential backoff retry logic, and token usage accounting.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = (
    "你是一个运行在 QQ 私聊环境中的私人智能助手。\n"
    "请遵循以下要求与风格：\n"
    "1. 始终使用自然、亲切、生动的简体中文进行交流。\n"
    "2. 解答问题应当条理清晰、内容详实充分，提供充足有用的背景和细节，避免只给出一两句话的过于简略的回答。\n"
    "3. 严禁使用任何 Markdown 格式（绝对不要使用 **加粗**、# 标题、* 列表星号、```代码块``` 等），必须输出为适合手机 QQ 私聊直接阅读的纯文本段落（可以使用 1. 2. 3. 或 - 等普通文本符号分点）。\n"
    "4. 除非用户在提问中明确要求提供链接、出处或网址，否则绝对不要主动附加或输出任何网址、URL 和参考链接。"
)


def get_active_system_prompt(custom_prompt: Optional[str] = None) -> str:
    """Load system prompt from custom string, prompts/system_prompt.txt, or default."""
    if custom_prompt and custom_prompt.strip():
        return custom_prompt.strip()

    prompt_path = Path("prompts/system_prompt.txt")
    if prompt_path.exists():
        try:
            content = prompt_path.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    return DEFAULT_SYSTEM_INSTRUCTION


class GroundingSource(BaseModel):
    title: str
    url: str


class GeminiResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    search_queries: List[str] = Field(default_factory=list)
    sources: List[GroundingSource] = Field(default_factory=list)
    model_used: str = ""


class BaseGeminiClient(ABC):
    @abstractmethod
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
        """Generate response from Gemini model."""
        pass


class GoogleGenAIClient(BaseGeminiClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-3.8-flash",
        default_thinking_level: str = "MEDIUM",
        default_max_output_tokens: int = 2048,
        system_instruction: Optional[str] = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.default_thinking_level = default_thinking_level.upper()
        self.default_max_output_tokens = default_max_output_tokens
        self.system_instruction = get_active_system_prompt(system_instruction)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Initialize the official google-genai Client
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Picks up GEMINI_API_KEY from environment automatically
            self.client = genai.Client()

    def _map_thinking_level(self, level: Optional[str]) -> types.ThinkingLevel:
        lvl = (level or self.default_thinking_level).upper().strip()
        # Strictly only LOW, MEDIUM, HIGH are supported.
        # MINIMAL is completely disallowed across all models and clamped to LOW.
        mapping = {
            "LOW": types.ThinkingLevel.LOW,
            "MEDIUM": types.ThinkingLevel.MEDIUM,
            "HIGH": types.ThinkingLevel.HIGH,
        }
        if lvl in ("MINIMAL", "NONE"):
            logger.info(f"[GeminiClient] Thinking level '{lvl}' is disallowed; clamped to 'LOW'.")
            return types.ThinkingLevel.LOW
        return mapping.get(lvl, types.ThinkingLevel.MEDIUM)

    def _build_contents(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> List[types.Content]:
        """Convert chat history dicts into google-genai types.Content objects."""
        contents: List[types.Content] = []
        num_msgs = len(messages)
        for idx, msg in enumerate(messages):
            role = "user" if msg.get("role") == "user" else "model"
            text = msg.get("content", "").strip()
            parts: List[types.Part] = []

            # If this is the last user message and images are present, attach images
            if idx == num_msgs - 1 and role == "user" and images:
                for img_data in images:
                    img_bytes = img_data.get("bytes")
                    mime = img_data.get("mime_type", "image/jpeg")
                    if img_bytes:
                        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

            if text:
                parts.append(types.Part.from_text(text=text))

            if parts:
                contents.append(
                    types.Content(
                        role=role,
                        parts=parts,
                    )
                )
        return contents

    def _extract_grounding(self, candidate: Any) -> tuple[List[str], List[GroundingSource]]:
        """Safely extract search queries and grounding source citations."""
        queries: List[str] = []
        sources: List[GroundingSource] = []
        seen_urls = set()

        if not candidate or not hasattr(candidate, "grounding_metadata") or not candidate.grounding_metadata:
            return queries, sources

        meta = candidate.grounding_metadata

        # Extract search queries
        if hasattr(meta, "web_search_queries") and meta.web_search_queries:
            queries = [str(q) for q in meta.web_search_queries if q]

        # Extract source chunks
        if hasattr(meta, "grounding_chunks") and meta.grounding_chunks:
            for chunk in meta.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    title = getattr(chunk.web, "title", "") or getattr(chunk.web, "domain", "") or "Web Source"
                    url = getattr(chunk.web, "uri", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append(GroundingSource(title=title.strip(), url=url.strip()))

        return queries, sources

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
        target_model = model or self.default_model
        if "2.5" in target_model.lower():
            raise ValueError(
                f"Gemini 2.5 系列模型 ({target_model}) 已被淘汰并停止支持。请使用 Gemini 3/4 新一代模型：gemini-3.5-flash-lite, gemini-3.8-flash 或 gemini-4-argon。"
            )

        contents = self._build_contents(messages, images=images)
        if not contents:
            return GeminiResponse(text="[无有效输入内容]", model_used=target_model)

        # Configure tools (Search Grounding)
        tools = []
        if enable_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Configure thinking: strictly only LOW, MEDIUM, HIGH are supported
        lvl_enum = self._map_thinking_level(thinking_level)
        thinking_cfg = types.ThinkingConfig(thinking_level=lvl_enum)

        max_tokens = max_output_tokens or self.default_max_output_tokens
        sys_inst = system_instruction if system_instruction is not None else self.system_instruction
        config = types.GenerateContentConfig(
            system_instruction=sys_inst,
            tools=tools if tools else None,
            thinking_config=thinking_cfg,
            max_output_tokens=max_tokens,
            temperature=0.2 if system_instruction is not None else 0.7,
        )

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Call Gemini async API with timeout
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=target_model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=self.timeout_seconds,
                )

                # Extract text
                resp_text = ""
                if response.text:
                    resp_text = response.text
                elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            resp_text += part.text

                # Extract token usage
                prompt_toks = 0
                cand_toks = 0
                total_toks = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_toks = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    cand_toks = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    total_toks = getattr(response.usage_metadata, "total_token_count", 0) or (prompt_toks + cand_toks)

                # Extract Grounding citations
                queries, sources = [], []
                if response.candidates:
                    queries, sources = self._extract_grounding(response.candidates[0])

                return GeminiResponse(
                    text=resp_text,
                    prompt_tokens=prompt_toks,
                    candidate_tokens=cand_toks,
                    total_tokens=total_toks,
                    search_queries=queries,
                    sources=sources,
                    model_used=target_model,
                )

            except asyncio.TimeoutError as te:
                last_err = te
                logger.warning(
                    f"[GeminiClient] Attempt {attempt}/{self.max_retries} timed out after {self.timeout_seconds}s"
                )
            except APIError as ae:
                last_err = ae
                # Check status code
                code = getattr(ae, "code", None)
                message = getattr(ae, "message", str(ae))
                logger.warning(
                    f"[GeminiClient] Attempt {attempt}/{self.max_retries} APIError (code={code}): {message}"
                )
                # Don't retry on client errors like 400 Bad Request or 401/403 Auth errors
                if code in (400, 401, 403, 404):
                    raise ae
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[GeminiClient] Attempt {attempt}/{self.max_retries} unexpected exception: {e}"
                )

            if attempt < self.max_retries:
                # Exponential backoff: 1s, 2s, 4s...
                delay = 1.5 ** attempt
                await asyncio.sleep(delay)

        # All retries failed
        if last_err:
            raise last_err
        raise RuntimeError("Failed to generate response after all retry attempts")
