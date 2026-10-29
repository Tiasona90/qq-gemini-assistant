"""
Chat Manager: Orchestrates user sessions, access control, concurrency,
message deduplication, multi-turn history, command parsing, and response formatting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from config import Settings
from db import Database
from gemini_client import BaseGeminiClient, GeminiResponse
from image_utils import download_image, compress_image

logger = logging.getLogger(__name__)


# Domains requiring mandatory search grounding
MANDATORY_SEARCH_KEYWORDS = {
    # Medical & medication
    "病", "药", "用药", "剂量", "症状", "治疗", "感染", "副作用", "医院", "胶囊", "处方",
    # Chemistry
    "化学", "化学式", "化合物", "反应式", "毒性", "溶剂", "催化剂",
    # Computer hardware
    "显卡", "CPU", "主板", "芯片", "驱动", "功耗", "架构", "GPU", "固件", "硬件参数", "台式机", "笔记本配置",
    # Real-time / recent events (2024 - 2026+)
    "最新", "最近", "今天", "明天", "今年", "2024", "2025", "2026", "多少钱", "价格", "新闻", "营业时间", "政策", "调整", "发布会"
}

# Domains requiring risk warnings
RISK_KEYWORDS = {
    "medical": ["吃药", "用药", "剂量", "生病", "症状", "确诊", "处方", "挂号", "输液"],
    "legal": ["起诉", "判刑", "法律责任", "立案", "劳动仲裁", "合同纠纷", "侵权"],
    "finance": ["股票", "基金", "理财", "投资", "炒币", "贷款", "做空", "牛市", "熊市"],
}

MODEL_ALIASES = {
    # Gemini 3.5 Flash Lite (经济省钱)
    "3.5": "gemini-3.5-flash-lite",
    "3.5flash": "gemini-3.5-flash-lite",
    "3.5-flash": "gemini-3.5-flash-lite",
    "3.5lite": "gemini-3.5-flash-lite",
    "3.5-flash-lite": "gemini-3.5-flash-lite",
    "3.5-lite": "gemini-3.5-flash-lite",
    "flashlite": "gemini-3.5-flash-lite",
    "lite": "gemini-3.5-flash-lite",
    # Gemini 3.8 Flash (默认主力)
    "3.8": "gemini-3.8-flash",
    "3.8flash": "gemini-3.8-flash",
    "3.8-flash": "gemini-3.8-flash",
    "3.8_flash": "gemini-3.8-flash",
    "flash3.8": "gemini-3.8-flash",
    # Gemini 4 Argon (旗舰高阶)
    "4": "gemini-4-argon",
    "4argon": "gemini-4-argon",
    "4-argon": "gemini-4-argon",
    "argon": "gemini-4-argon",
    "gemini4": "gemini-4-argon",
    "gemini-4": "gemini-4-argon",
    "gemini-4-argon": "gemini-4-argon",
    # Note: Gemini 2.5 series has been deprecated and removed.
}

SUPPORTED_MODELS = {
    "gemini-3.5-flash-lite": "经济轻量型（极速省钱、消耗最低）",
    "gemini-3.8-flash": "默认主力型（日常通用平衡推荐）",
    "gemini-4-argon": "旗舰顶配型（能力最强、最聪明、适合复杂深度推理）",
}


def normalize_model_name(name: str) -> str:
    cleaned = name.strip()
    low = cleaned.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    stripped = re.sub(r"^gemini-?", "", low)
    if stripped in MODEL_ALIASES:
        return MODEL_ALIASES[stripped]
    return cleaned


SUMMARY_SYSTEM_INSTRUCTION = (
    "你是一个通用的专业对话记忆提炼助手。你的任务是将长篇历史对话提炼合并为一段客观、准确、信息密集的背景摘要。\n"
    "通用提炼原则：\n"
    "1. 广泛适用于各种领域（技术开发、软硬件排查、数码选购、学术研讨、生活常识、情感倾诉、医疗健康、创作策划等所有主题）。\n"
    "2. 深度提炼核心事实：提取用户核心诉求、约束限制与个人偏好、关键专有名词、软硬件型号、具体参数与数字、人物与关系、达成的决策共识或已尝试的操作步骤。\n"
    "3. 剔除所有冗余闲聊、寒暄客套、重复确认与过渡语句。\n"
    "4. 篇幅目标基准约为 300 字，但若该段对话信息量极大、细节复杂，在你认为 300 字不足以完整保留所有核心事实时，允许根据实际需要适当扩充字数，首要目标是保证关键事实与脉络完整无遗漏。\n"
    "5. 纯文本段落输出，不要使用任何 Markdown 格式或多余说明。"
)


class ChatManager:
    def __init__(self, settings: Settings, db: Database, client: BaseGeminiClient) -> None:
        self.settings = settings
        self.db = db
        self.client = client

        # Global concurrency semaphore (default max 2)
        self.concurrency_semaphore = asyncio.Semaphore(self.settings.max_gemini_concurrency)

        # Per-user FIFO lock
        self._user_locks: Dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

        # Start time for uptime reporting
        self.start_time = time.time()

        # In-memory deduplication cache (LRU-like set)
        self._seen_msg_ids: set[str] = set()

        # Track active background compression tasks per user
        self._compressing_users: set[str] = set()

        # Staged pending images per user: user_id -> {"images": [...], "timestamp": float}
        self._pending_images: Dict[str, Dict[str, Any]] = {}

    async def _effective_user_settings(self, user_id: str) -> Dict[str, Any]:
        overrides = await self.db.get_user_settings(user_id) or {}
        defaults = {
            "model": self.settings.gemini_model,
            "thinking_level": self.settings.default_thinking_level,
            "context_messages": self.settings.default_context_messages,
            "search_enabled": self.settings.enable_search_default,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        return {key: overrides.get(key) if overrides.get(key) is not None else value
                for key, value in defaults.items()}

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    def _should_trigger_search(self, query: str, user_search_enabled: Optional[bool]) -> bool:
        """
        Determine if Google Search Grounding should be enabled for this query.
        """
        # If user explicitly turned off search, respect user preference unless high risk
        if user_search_enabled is False:
            return False

        # If user explicitly turned on search or default is true:
        # Check mandatory keywords
        lower_q = query.lower()
        for kw in MANDATORY_SEARCH_KEYWORDS:
            if kw.lower() in lower_q:
                return True

        # Check for year references >= 2024
        if re.search(r"202[4-9]|203\d", query):
            return True

        # Check for price / store hours / policy questions
        if any(w in query for w in ["多少钱", "价格", "明天", "昨天", "最新", "营业", "开门", "放假"]):
            return True

        # If default is true and query seems like an informational query, enable search
        return user_search_enabled if user_search_enabled is not None else self.settings.enable_search_default

    def _strip_markdown(self, text: str) -> str:
        """Strip markdown syntax to ensure clean plain text for QQ messaging."""
        # Remove markdown images: ![alt](url)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # Remove code blocks ```lang ... ```
        text = re.sub(r"```(?:\w+)?\n?", "", text)
        # Remove bold/italic: **text**, *text*, __text__, _text_
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"__(.*?)__", r"\1", text)
        text = re.sub(r"_(.*?)_", r"\1", text)
        # Remove markdown headers: # Header
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        # Remove blockquotes: > quote
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
        # Remove inline code backticks: `code`
        text = re.sub(r"`(.*?)`", r"\1", text)
        return text.strip()

    def _format_reply(self, response: GeminiResponse, user_query: str) -> str:
        """
        Format Gemini response into concise QQ plain text.
        Strips markdown completely.
        Only attaches reference links if explicitly requested by the user.
        """
        text = self._strip_markdown(response.text)

        # Do NOT actively attach reference links unless explicitly requested
        wants_links = any(k in user_query.lower() for k in ["链接", "网址", "来源", "出处", "url", "link", "参考"])
        if wants_links and response.sources:
            citations = []
            for idx, s in enumerate(response.sources[:4], 1):
                clean_title = s.title.replace("\n", " ").strip()
                citations.append(f"{idx}. {clean_title}\n   {s.url}")
            if citations:
                text += "\n\n【参考来源】\n" + "\n".join(citations)

        # Append risk disclaimer if relevant
        for category, kws in RISK_KEYWORDS.items():
            if any(k in user_query for k in kws):
                text += "\n\n【注意】以上内容由 AI 生成，仅供参考，不构成专业医疗、法律或金融投资建议。重要事项请咨询专业人员。"
                break

        return text.strip()

    async def _get_help_text(self, user_id: str, topic: str = "") -> str:
        """Readable full guide and local, read-only topic help."""
        cfg = await self._effective_user_settings(user_id)
        thinking_labels = {"LOW": "低", "MEDIUM": "中", "HIGH": "高"}
        level = str(cfg["thinking_level"]).upper()
        context = cfg["context_messages"]
        sections = {
            "基础": (
                "【1. 基础操作】\n"
                "//help：查看完整帮助和你当前的配置。\n"
                "//help 模型：只看某一类帮助；其他分类见文末。\n"
                "//status：查看运行时间、用量、会话摘要和待问图片。\n"
                "//reset：开始新话题，清空你的聊天记录、会话摘要和待问图片。\n"
                "  不会恢复模型等设置，也不会清零累计用量。\n"
                "//成本：查看个人及全局请求数、Token 和估算费用。\n"
                "  Token 是模型处理文字等内容的计量单位，不等于字数；费用为程序估算，以服务商账单为准。\n"
                "中文别名：//帮助、//状态、//重置；//清空 等同重置。\n"
                "//花费、//费用、//用量、//统计、//token 均可查看成本。"
            ),
            "模型": (
                "【2. 模型：选择由哪个模型回答】\n"
                "//模型：查看当前模型。\n"
                "//模型 3.5：切换到轻量档 gemini-3.5-flash-lite。\n"
                "//模型 3.8：切换到主力档 gemini-3.8-flash。\n"
                "//模型 4argon：切换到旗舰档 gemini-4-argon。\n"
                "这是项目预设的三档，也可填写完整模型名；能否调用取决于 API 的实际支持和权限。\n"
                "想用口语表达，可发送：\n"
                "//换个笨一点的模型 → 降一档，已在轻量档则保持。\n"
                "//换个更好的模型 → 升一档，已在旗舰档则保持。\n"
                "//换成最便宜的模型 → 直接选轻量档。\n"
                "//换成最强的模型 → 直接选旗舰档。\n"
                "填写模型名时，在“模型”后留一个空格。"
            ),
            "思考": (
                "【3. 思考：调整当前模型的思考强度】\n"
                "//低思考：LOW，适合简单问答。\n"
                "//中思考：MEDIUM，中等强度。\n"
                "//高思考：HIGH，给复杂问题更多思考空间，可能更慢、消耗更多。\n"
                "//思考强度更高一点：在当前基础上升一档。\n"
                "//思考再低一点：在当前基础上降一档。\n"
                "只支持低、中、高三档；“关闭思考”也会按最低档 LOW 处理。\n"
                "调思考不会换模型；换模型也不会自动改思考强度。"
            ),
            "搜索": (
                "【4. 搜索：允许或关闭实时联网检索】\n"
                "//搜索：查看当前搜索设置。\n"
                "//搜索 开：允许联网搜索，由程序按问题触发。\n"
                "//搜索 关：明确关闭联网搜索，回答依靠模型已有知识。\n"
                "开关前留一个空格。英文写法：/search on、/search off。\n"
                "关闭后，最新信息可能不准确；仅开启搜索不代表每条回复都有来源链接。\n"
                "需要来源时，在提问中写“请附上来源链接”。"
            ),
            "记忆": (
                "【5. 记忆：调整普通历史回带量】\n"
                "1 轮 = 你问一次 + 助手答一次 = 2 条消息。\n"
                "//5轮对话记录：设置为 5 轮，即 10 条消息。\n"
                "//记录8条历史：设置为 8 条消息，即 4 轮。\n"
                "//多记2轮对话：在当前设置上增加 2 轮。\n"
                "//减少2轮对话：在当前设置上减少 2 轮。\n"
                "范围为 0～50 条消息，最多相当于 25 轮；携带越多，输入消耗通常越多。\n"
                "注意：连续会话还有独立的动态记忆，会携带未压缩对话及已有摘要；上面的数字不代表每次请求的硬上限。\n"
                "//0轮对话记录 只关闭普通历史回带，不会删除记录或清除连续会话。\n"
                "想彻底开始新话题，请用 //reset。"
            ),
            "输出": (
                "【6. 输出：设置单次回答的长度上限】\n"
                "//输出长度 2048：最大输出设置为 2048 Token。\n"
                "//输出长度 4096：给长回答留更多空间。\n"
                "明确数字设置支持 100～8192 Token，超出会自动限制在此范围内。\n"
                "这是上限，不是必须输出的字数；上限太小可能让回答中途截断。\n"
                "想控制表达方式，可直接提问“请简短回答”或“请详细解释”。"
            ),
            "图片": (
                "【7. 图片：直接发图，不需要指令】\n"
                "方法一：图片和问题一起发，例如“请解释这张图”。\n"
                "方法二：先发图片，收到接收提示后，再发送问题。\n"
                "单独发送的图片暂存 15 分钟，供下一条普通文字提问使用；多张图可同一条消息发送。\n"
                "再次单独发图会替换之前暂存的那批；已用于提问后，再次分析请重发图片。\n"
                "图片会处理后交给模型，请尽量发送清晰原图。\n"
                "//status 可查看是否有待问图片；//reset 会清除待问图片。"
            ),
            "用法": (
                "【8. 指令怎么发】\n"
                "普通聊天：直接发文字；修改设置：单独发送以 // 开头的指令。\n"
                "一次只改一项最清楚；也支持组合，例如：\n"
                "//把思考调成中，保留5轮对话记录\n"
                "口语指令优先本地识别，复杂表达可能调用 AI 提取，产生额外等待和用量；请核对返回的变更结果。\n"
                "解析失败时，改用本帮助中的明确写法。\n"
                "单斜杠兼容：/help、/帮助、/?、/reset、/重置、/status、/状态、/model、/模型、/search、/搜索。\n"
                "例如 /model 3.8；成本、思考、记忆、输出和口语调节请用双斜杠 //。\n"
                "模型、思考、搜索、记忆、输出设置只影响你的账号，保存后下次提问生效，重启后保留。"
            ),
        }
        current = (
            "【你当前的配置】\n"
            f"模型：{cfg['model']}\n"
            f"思考强度：{thinking_labels.get(level, level)}（{level}）\n"
            f"普通历史回带：{context} 条消息（相当于 {context / 2:g} 轮对话）\n"
            f"最大输出：{cfg['max_output_tokens']} Token\n"
            f"实时搜索：{'开启' if cfg['search_enabled'] else '关闭'}"
        )
        topics = "基础 / 模型 / 思考 / 搜索 / 记忆 / 输出 / 图片 / 用法 / 当前"
        navigation = f"【按分类查看】\n发送 //help 分类名，例如 //help 图片\n可选：{topics}"
        aliases = {
            "基础": "基础", "命令": "基础", "指令": "基础", "成本": "基础", "用量": "基础",
            "模型": "模型", "model": "模型", "思考": "思考", "thinking": "思考",
            "搜索": "搜索", "search": "搜索", "记忆": "记忆", "历史": "记忆", "context": "记忆",
            "输出": "输出", "长度": "输出", "tokens": "输出", "图片": "图片", "image": "图片",
            "用法": "用法", "当前": "当前", "配置": "当前", "settings": "当前",
        }
        topic = topic.strip().lower()
        if topic in ("", "全部", "完整", "all"):
            intro = (
                "【Gemini 智能助手 · 指令帮助】\n"
                "直接发文字就能聊天，发图片也能提问。下面的指令可整行复制，每次单独发送。\n"
                "只想查一项？例如发送 //help 模型 或 //help 图片。"
            )
            return "\n\n".join([intro, current, *sections.values(), navigation])
        key = aliases.get(topic)
        if key == "当前":
            return current + "\n\n" + navigation
        if key:
            return sections[key] + "\n\n" + current + "\n\n" + navigation
        return f"未找到帮助分类“{topic}”。\n{navigation}\n发送 //help 查看完整帮助。"

    @staticmethod
    def _parse_thinking_command(text: str, current: str) -> Optional[str]:
        """Resolve explicit levels before relative changes, without splitting 强度.

        Atomic grouping prevents regex backtracking from reducing 思考强度 to
        思考 and then mistaking 强 for a request to increase the level.
        """
        head = r"(?>思考(?:强度|深度|模式|等级|级别)?|思维(?:强度|深度|模式)?|推理(?:强度|深度|模式)?|thinking(?:[ _-]*level)?)"
        level = r"(?:MINIMAL|MEDIUM|HIGH|LOW|极低|最小|关闭|关掉|中等|低|中|高|无)"
        level_end = r"(?![a-zA-Z]|一点|一些|点|些)"
        mapping = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "MINIMAL": "LOW",
                   "低": "LOW", "中": "MEDIUM", "中等": "MEDIUM", "高": "HIGH",
                   "极低": "LOW", "最小": "LOW", "关闭": "LOW", "关掉": "LOW", "无": "LOW"}

        # Explicit targets, including 从高调到低; source levels are not targets.
        setter = r"(?:(?:设置|设|修改|调整|调节|改|调|切换)\s*(?:为|成|到|至)|(?:为|成|到|至|是|=|:|：)|(?:设置|设))"
        absolute = re.search(
            rf"{head}\s*(?:从\s*{level}\s*)?{setter}\s*({level}){level_end}", text, re.I)
        if absolute:
            return mapping[absolute.group(1).upper()]
        def step(direction: int) -> str:
            ladder = ["LOW", "MEDIUM", "HIGH"]
            index = ladder.index(current) if current in ladder else 0
            return ladder[max(0, min(index + direction, len(ladder) - 1))]

        # Verb-first requests must precede short levels: 降低思考 contains 低思考.
        if re.search(rf"(?:降低|减弱|弱化|减少|调低|下调)\s*{head}", text, re.I):
            return step(-1)
        if re.search(rf"(?:提高|增强|加强|增加|调高|上调)\s*{head}", text, re.I):
            return step(1)

        # Short forms: 低思考, 高思考; and 思考强度 LOW / 思考强度中.
        for pattern in (rf"({level})\s*{head}", rf"{head}\s*({level}){level_end}"):
            absolute = re.search(pattern, text, re.I)
            if absolute:
                return mapping[absolute.group(1).upper()]

        prefix = r"(?:(?:再|更|稍微|稍|适当|给我|帮我)\s*)?(?:(?:调整|调节|调)\s*)?(?:(?:再|更|稍微)\s*)?"
        up = r"(?:提高|增强|加强|增加|升高|高|强|多|升|加)"
        down = r"(?:降低|减弱|弱化|减少|低|弱|少|降|减)"
        direction = None
        if re.search(rf"{head}\s*{prefix}{down}|(?:降低|减弱|弱化|减少)\s*{head}", text, re.I):
            direction = -1
        elif re.search(rf"{head}\s*{prefix}{up}|(?:提高|增强|加强|增加)\s*{head}", text, re.I):
            direction = 1
        if direction is not None:
            return step(direction)
        return None

    def _parse_command_fast(self, text: str, user_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fast local pattern extraction for common parameter & fuzzy commands.
        Extracts thinking_level, context_messages, search_enabled, model, max_output_tokens.
        Supports both absolute settings and relative context-aware adjustments.
        """
        result: Dict[str, Any] = {}
        t = text.strip()
        cfg = user_cfg or {}
        cur_think = str(cfg.get("thinking_level") or self.settings.default_thinking_level).upper()
        cur_model = str(cfg.get("model") or self.settings.gemini_model).lower()
        cur_ctx = int(cfg["context_messages"] if cfg.get("context_messages") is not None else self.settings.default_context_messages)

        # Match full thinking nouns before looking for relative direction words.
        is_thinking_cmd = bool(re.search(r"思考|思维|推理|深度|想一想|thinking", t, re.I))
        thinking_level = self._parse_thinking_command(t, cur_think)
        if thinking_level is not None:
            result["thinking_level"] = thinking_level

        # 2. Context messages / rounds (Relative or Absolute)
        cn_num_map = {
            "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            "十一": 11, "十二": 12, "十五": 15, "二十": 20,
        }
        def _to_num(val_str: str) -> Optional[int]:
            v = val_str.strip()
            if v.isdigit():
                return int(v)
            return cn_num_map.get(v)

        rel_more = re.search(r"(?:多|再|额外)(?:记|增加|保留|提供)\s*(\d+|[一二两三四五六七八九十]+)\s*轮", t)
        rel_less = re.search(r"(?:少|减少)\s*(\d+|[一二两三四五六七八九十]+)\s*轮", t)
        if rel_more:
            r = _to_num(rel_more.group(1)) or 1
            result["context_messages"] = min(50, cur_ctx + r * 2)
        elif rel_less:
            r = _to_num(rel_less.group(1)) or 1
            result["context_messages"] = max(0, cur_ctx - r * 2)
        else:
            rounds_match = re.search(
                r"(?:(?:把|将)?(?:历史|对话|上下文)?(?:记录|深度)?(?:设置(?:为|成)?|修改(?:为|成)?|调整(?:为|成)?|改成?|保留|记录|携带|提供))\s*(\d+|[一二两三四五六七八九十]+)\s*轮",
                t
            )
            if not rounds_match:
                rounds_match = re.search(r"(\d+|[一二两三四五六七八九十]+)\s*轮(?:对话)?(?:记录)?", t)

            if rounds_match:
                rounds = _to_num(rounds_match.group(1))
                if rounds is not None:
                    result["context_messages"] = max(0, min(rounds * 2, 50))
            else:
                msgs_match = re.search(
                    r"(?:(?:把|将)?(?:历史|对话|上下文)?(?:记录|深度)?(?:设置(?:为|成)?|修改(?:为|成)?|调整(?:为|成)?|改成?|保留|记录|携带|提供))\s*(\d+|[一二两三四五六七八九十]+)\s*条",
                    t
                )
                if not msgs_match:
                    msgs_match = re.search(r"(\d+|[一二两三四五六七八九十]+)\s*条(?:历史)?(?:消息)?(?:记录)?", t)
                if msgs_match:
                    msgs_val = _to_num(msgs_match.group(1))
                    if msgs_val is not None:
                        result["context_messages"] = max(0, min(msgs_val, 50))

        # 3. Search Grounding toggle
        if re.search(r"(?:(?:开启|打开|启用)\s*(?:实时)?搜索|(?:实时)?搜索\s*(?:设置(?:为|成)?|调(?:为|成)?)?\s*(?:开启|打开|启用|开|on|true))", t, re.I):
            result["search_enabled"] = True
        elif re.search(r"(?:(?:关闭|关掉|禁用)\s*(?:实时)?搜索|(?:实时)?搜索\s*(?:设置(?:为|成)?|调(?:为|成)?)?\s*(?:关闭|关掉|禁用|关|off|false))", t, re.I):
            result["search_enabled"] = False

        # 4. Model switching (Relative or Absolute)
        verbs = r"(?:换(?:成|到)?|用|切(?:换)?(?:到|成)?|改(?:成|用)?|选|降(?:级)?|升(?:级)?)"
        down_words = r"(?:便宜|省钱|轻量|经济|低级|基础|低配|笨|弱|差|低|降级)"
        up_words = r"(?:好|强|聪明|高级|顶配|旗舰|高|升级)"

        is_downgrade = bool(re.search(rf"(?:{verbs}?(?:一个|个)?(?:更|最)?{down_words}(?:一点|一些|点)?(?:的)?(?:模型|版本)?|降级(?:模型)?|省钱模式)", t))
        is_upgrade = bool(re.search(rf"(?:{verbs}?(?:一个|个)?(?:更|最)?{up_words}(?:一点|一些|点)?(?:的)?(?:模型|版本)?|升级(?:模型)?)", t))

        # Disambiguate: if the user explicitly meant thinking, don't trigger model upgrade/downgrade unless model is also mentioned
        if is_thinking_cmd and not bool(re.search(r"模型|model|版本", t)):
            is_downgrade = False
            is_upgrade = False

        if is_downgrade:
            if any(w in t for w in ["最便宜", "最省钱", "最轻量", "最低级", "最低配", "省钱模式"]):
                result["model"] = "gemini-3.5-flash-lite"
            elif "4" in cur_model or "argon" in cur_model:
                result["model"] = "gemini-3.8-flash"
            else:
                result["model"] = "gemini-3.5-flash-lite"
        elif is_upgrade:
            if any(w in t for w in ["最好", "最强", "最顶配", "最聪明", "最高级", "旗舰"]):
                result["model"] = "gemini-4-argon"
            elif "3.5" in cur_model or "lite" in cur_model:
                result["model"] = "gemini-3.8-flash"
            else:
                result["model"] = "gemini-4-argon"
        elif any(w in t for w in ["默认模型", "主力模型", "标准模型", "平衡模型", "恢复默认模型"]):
            result["model"] = "gemini-3.8-flash"
        if "model" not in result:
            model_match = re.search(
                r"(?:(?:把|将)?(?:(?:设置|切换|修改|改用|换用|选用|使用)?\s*模型\s*(?:设置|切换|修改|为|成|到)?))\s*([a-zA-Z0-9\.\-_]+)",
                t,
                re.I
            )
            if not model_match:
                model_match = re.search(r"([a-zA-Z0-9\.\-_]+)\s*模型", t, re.I)

            if not model_match and (t.startswith("模型") or t.lower().startswith("model")):
                parts = t.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    raw_m = parts[1].strip()
                    result["model"] = normalize_model_name(raw_m)
            elif model_match:
                cand = model_match.group(1).strip()
                if not cand.isdigit() and cand.lower() not in ("on", "off", "true", "false", "low", "high", "medium"):
                    result["model"] = normalize_model_name(cand)

        # 5. Max output tokens (输出长度)
        len_match = re.search(
            r"(?:(?:把|将)?(?:最大)?(?:输出|回复)?长度\s*(?:设置(?:为|成)?|修改(?:为|成)?|调整(?:为|成)?|改成?)?|max_tokens?\s*(?:为|成)?)\s*(\d+)",
            t,
            re.I
        )
        if len_match:
            result["max_output_tokens"] = max(100, min(int(len_match.group(1)), 8192))

        return result

    async def _apply_extracted_settings(self, user_id: str, parsed: Dict[str, Any]) -> str:
        """Apply extracted configuration dictionary to user settings and build user confirmation text."""
        applied = []
        new_lvl = parsed.get("thinking_level")
        if new_lvl and str(new_lvl).upper() in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
            lvl_str = str(new_lvl).upper()
            if lvl_str == "MINIMAL":
                lvl_str = "LOW"  # Strictly disallowed, clamp to LOW
            await self.db.update_user_settings(user_id, thinking_level=lvl_str)
            applied.append(f"思考强度={lvl_str}")

        new_ctx = parsed.get("context_messages")
        if type(new_ctx) is int and 0 <= new_ctx <= 50:
            rounds_str = f"（对应 {new_ctx // 2} 轮对话）" if new_ctx % 2 == 0 and new_ctx > 0 else ""
            await self.db.update_user_settings(user_id, context_messages=new_ctx)
            applied.append(f"历史记忆={new_ctx} 条消息 {rounds_str}".strip())

        new_search = parsed.get("search_enabled")
        if isinstance(new_search, bool):
            await self.db.update_user_settings(user_id, search_enabled=new_search)
            applied.append(f"联网搜索={'开启' if new_search else '关闭'}")

        new_model = parsed.get("model")
        if new_model and isinstance(new_model, str) and len(new_model) < 64:
            norm_model = normalize_model_name(new_model)
            if "2.5" in norm_model.lower():
                return "【配置提示】系统已取消对 Gemini 2.5 旧模型的支持。请选择新一代模型：gemini-3.5-flash-lite、gemini-3.8-flash 或 gemini-4-argon。"
            await self.db.update_user_settings(user_id, model=norm_model)
            applied.append(f"专属模型={norm_model}")

        new_tokens = parsed.get("max_output_tokens")
        if type(new_tokens) is int and 100 <= new_tokens <= 16384:
            await self.db.update_user_settings(user_id, max_output_tokens=new_tokens)
            applied.append(f"最大输出={new_tokens} tokens")

        if applied:
            intent = str(parsed.get("intent") or "").strip()
            msg = "【智能配置已生效】\n"
            if intent:
                msg += f"💡 识别意图：{intent}\n"
            msg += "已为你的账号成功应用：" + "，".join(applied)
            return msg
        return ""

    async def compress_session_history(self, user_id: str) -> Optional[str]:
        """
        Compress older messages in active continuous session into a unified background summary
        using gemini-3.5-flash-lite, preserving recent 4 rounds (8 messages) verbatim.
        Returns the new summary text if generated, or None.
        """
        uid = str(user_id).strip()
        if uid in self._compressing_users:
            return None

        self._compressing_users.add(uid)
        try:
            session_info = await self.db.get_session_memory(uid)
            compressed_msg_id = session_info.get("compressed_msg_id", 0)
            prior_summary = session_info.get("summary", "")

            uncompressed = await self.db.get_uncompressed_messages(uid, compressed_msg_id)
            threshold = self.settings.session_compression_threshold_rounds * 2  # e.g. 20 msgs (10 rounds)
            keep_count = self.settings.session_keep_recent_rounds * 2  # e.g. 8 msgs (4 rounds)

            if len(uncompressed) < threshold:
                return None

            to_compress = uncompressed[:-keep_count]
            if not to_compress:
                return None

            new_compressed_id = to_compress[-1]["id"]
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in to_compress])

            prompt = "请将以下对话历史整理提炼为连贯的背景摘要。\n\n"
            if prior_summary:
                prompt += f"【已有前文摘要】：\n{prior_summary}\n\n"
            prompt += f"【需要合并的新增历史对话】：\n{history_text}\n\n请输出合并更新后的背景摘要："

            try:
                resp = await self.client.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    model="gemini-3.5-flash-lite",
                    thinking_level="LOW",
                    enable_search=False,
                    max_output_tokens=1024,
                    system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
                )
            except Exception:
                resp = await self.client.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.settings.gemini_model,
                    thinking_level="LOW",
                    enable_search=False,
                    max_output_tokens=1024,
                    system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
                )

            new_summary = self._strip_markdown(resp.text.strip())
            await self.db.update_session_memory(
                user_id=uid,
                summary=new_summary,
                compressed_msg_id=new_compressed_id,
            )
            logger.info(
                f"[ChatManager] Successfully compressed {len(to_compress)} messages into summary for user={uid} "
                f"(up to msg_id={new_compressed_id})"
            )
            return new_summary
        except Exception as e:
            logger.error(f"[ChatManager] Error compressing session history for user={uid}: {e}")
            return None
        finally:
            self._compressing_users.discard(uid)

    async def handle_command(self, user_id: str, raw_text: str) -> Optional[str]:
        """
        Handle built-in commands starting with '//' or '/'.
        Returns reply text if handled, or None if it's a regular chat message.
        """
        text = raw_text.strip()

        # Route full/topic help before fuzzy extraction, for either slash prefix.
        help_match = re.fullmatch(r"/{1,2}(?:help|帮助|[?？])(?:\s+(.*))?", text, re.I | re.S)
        if help_match:
            return await self._get_help_text(user_id, help_match.group(1) or "")

        # 1. Parameter & Fuzzy commands starting with '//' (checked first!)
        if text.startswith("//"):
            param_text = text[2:].strip()

            # //help or //帮助
            if param_text.lower() in ("help", "帮助", "?", "？"):
                return await self._get_help_text(user_id)

            # //reset or //重置
            if param_text.lower() in ("reset", "重置", "清空"):
                count = await self.db.clear_history(user_id)
                await self.db.clear_session_memory(user_id)
                self._pending_images.pop(user_id, None)
                return f"【会话已重置】已清空你的历史对话记录（共清理 {count} 条消息）。"

            # //status or //状态
            if param_text.lower() in ("status", "状态", "运行状态"):
                uptime_sec = int(time.time() - self.start_time)
                hrs = uptime_sec // 3600
                mins = (uptime_sec % 3600) // 60
                secs = uptime_sec % 60
                user_cfg = await self._effective_user_settings(user_id)
                cur_model = user_cfg.get("model") or self.settings.gemini_model
                stats = await self.db.get_usage_summary(user_id)
                session_mem = await self.db.get_session_memory(user_id)
                summary_status = f"已生成 ({len(session_mem.get('summary', ''))}字)" if session_mem.get("summary") else "无 (未触发压缩)"
                pending_img_status = "有 (待发送提问)" if user_id in self._pending_images else "无"

                return (
                    "【智能助手运行状态】\n"
                    "----------------------\n"
                    f"- 运行时间: {hrs}小时{mins}分{secs}秒\n"
                    f"- 全局并发限制: {self.settings.max_gemini_concurrency}\n"
                    f"- 数据库状态: 正常 (WAL 模式)\n"
                    f"- 你的专属模型: {cur_model}\n"
                    f"- 会话记忆摘要: {summary_status}\n"
                    f"- 暂存待问图片: {pending_img_status}\n"
                    f"- 你的提问次数: {stats['user']['requests']} 次\n"
                    f"- 你的总消耗 Token: {stats['user']['total_tokens']}\n"
                    f"- 全局总请求数: {stats['global']['requests']} 次"
                )

            # //模型 or //model: Dedicated per-user model switching
            if param_text.startswith("模型") or param_text.lower().startswith("model"):
                parts = param_text.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    new_model = normalize_model_name(parts[1].strip())
                    await self.db.update_user_settings(user_id, model=new_model)
                    return f"【专属模型已更新】你当前的独立模型已设置为: {new_model}"
                else:
                    user_cfg = await self._effective_user_settings(user_id)
                    cur_model = user_cfg.get("model") or self.settings.gemini_model
                    return (
                        f"【你的当前专属模型】: {cur_model}\n"
                        "切换命令示例: //模型 gemini-3.8-flash 或 //模型 gemini-3.5-flash-lite\n"
                        "提示：每个用户的模型设置各自独立记录，互不影响。"
                    )

            # //搜索 or //search
            if param_text.startswith("搜索") or param_text.lower().startswith("search"):
                parts = param_text.split(maxsplit=1)
                arg = parts[1].strip().lower() if len(parts) > 1 else ""
                if arg in ("开", "开启", "on", "1", "true"):
                    await self.db.update_user_settings(user_id, search_enabled=True)
                    return "【实时搜索】已开启。遇到需要最新信息或专业领域的问题将自动执行 Google 搜索检索。"
                elif arg in ("关", "关闭", "off", "0", "false"):
                    await self.db.update_user_settings(user_id, search_enabled=False)
                    return "【实时搜索】已关闭。模型将仅依靠自带训练知识回答。"
                else:
                    user_cfg = await self._effective_user_settings(user_id)
                    cur_search = user_cfg.get("search_enabled", self.settings.enable_search_default)
                    return f"【实时搜索当前状态】: {'开启' if cur_search else '关闭'}\n用法: //搜索 开 或 //搜索 关"

            # Cost & usage statistics
            if param_text in ("成本", "花费", "费用", "token", "用量", "统计"):
                stats = await self.db.get_usage_summary(user_id)
                u = stats["user"]
                g = stats["global"]

                # Price calculations (Gemini 3.8 / 3.5 Flash developer tier estimation)
                # Input: $0.075 / 1M tokens ($0.000000075 / token)
                # Output: $0.30 / 1M tokens ($0.00000030 / token)
                u_cost_usd = (u["prompt_tokens"] * 0.000000075) + (u["candidate_tokens"] * 0.00000030)
                u_cost_cny = u_cost_usd * 7.25

                g_cost_usd = (g["prompt_tokens"] * 0.000000075) + (g["candidate_tokens"] * 0.00000030)
                g_cost_cny = g_cost_usd * 7.25

                return (
                    "【成本与 Token 统计】\n"
                    "----------------------\n"
                    "【个人累计用量】\n"
                    f"- 提问次数: {u['requests']} 次\n"
                    f"- 输入 Token: {u['prompt_tokens']}\n"
                    f"- 输出 Token: {u['candidate_tokens']}\n"
                    f"- 总计 Token: {u['total_tokens']}\n"
                    f"- 搜索次数: {u['search_queries']} 次\n"
                    f"- 估算花费: ${u_cost_usd:.6f} USD (约 {u_cost_cny:.4f} 元)\n"
                    "\n"
                    "【系统全局用量】\n"
                    f"- 全局请求: {g['requests']} 次 | 总 Token: {g['total_tokens']}\n"
                    f"- 全局估算: ${g_cost_usd:.6f} USD (约 {g_cost_cny:.4f} 元)\n"
                    "----------------------\n"
                    "提示：当前使用付费 API，按实际 Token 用量实时计费，无免费层每日频次限制。"
                )

            # Fast local pattern extraction (with user current state for relative adjustments)
            user_cfg = await self._effective_user_settings(user_id)
            fast_parsed = self._parse_command_fast(param_text, user_cfg)
            if fast_parsed:
                reply = await self._apply_extracted_settings(user_id, fast_parsed)
                if reply:
                    return reply

            # Fuzzy natural language command parsing via AI (using cheap / fast extraction)
            logger.info(f"[ChatManager] Triggering AI fuzzy command extraction for: //{param_text}")
            try:
                cur_model = user_cfg.get("model") or self.settings.gemini_model
                cur_thinking = user_cfg.get("thinking_level") or self.settings.default_thinking_level
                cur_ctx = user_cfg.get("context_messages", self.settings.default_context_messages)
                cur_search = user_cfg.get("search_enabled", self.settings.enable_search_default)
                cur_tokens = user_cfg.get("max_output_tokens") or self.settings.max_output_tokens

                extraction_prompt = (
                    "你是一个智能指令分析与配置解析引擎。请深入理解用户的口语化/模糊指令，推断其真实意图，并以严格合法的单个 JSON 格式输出，不要包含任何 markdown 标记、代码块或多余解释说明。\n\n"
                    "【项目支持的 Gemini 模型梯队】\n"
                    "- gemini-3.5-flash-lite : 经济型，最轻量、极速省钱、消耗最低、低配/基础档次\n"
                    "- gemini-3.8-flash : 主力推荐，日常通用平衡型、默认标准档次\n"
                    "- gemini-4-argon : 旗舰顶配，能力最强、最聪明、适合复杂深度推理的高阶档次\n\n"
                    "【该用户当前配置状态】\n"
                    f"- 当前模型: {cur_model}\n"
                    f"- 当前思考强度: {cur_thinking} (可选梯度: LOW < MEDIUM < HIGH)\n"
                    f"- 当前上下文记忆: {cur_ctx} 条消息 (对应 {cur_ctx // 2} 轮对话)\n"
                    f"- 当前实时搜索: {'开启' if cur_search else '关闭'}\n"
                    f"- 当前最大输出长度: {cur_tokens} tokens\n\n"
                    "【自然语言与模糊指令语义理解规范】\n"
                    "1. 模型降级/经济意图（低级一点、差一点、低配、便宜点、省钱、轻量、简单、降级、基础模型等）：\n"
                    "   - 若当前为 gemini-4-argon 则降为 gemini-3.8-flash；若当前为 gemini-3.8-flash 则降为 gemini-3.5-flash-lite；若已是 gemini-3.5-flash-lite 则保持。\n"
                    "   - 若明确说“最便宜/最低级/最轻量/最低配”则直接降为 gemini-3.5-flash-lite。\n"
                    "2. 模型升级/高阶意图（高级一点、好一点、更强、更聪明、顶配、旗舰、高阶、升级模型等）：\n"
                    "   - 若当前为 gemini-3.5-flash-lite 则升为 gemini-3.8-flash；若当前为 gemini-3.8-flash 则升为 gemini-4-argon；若已是 gemini-4-argon 则保持。\n"
                    "   - 若明确说“最好/最强/最聪明/顶配”则直接升为 gemini-4-argon。\n"
                    "3. 思考强度意图（想一想、深入点、多动动脑、别想了、简单回答、少思考等）：\n"
                    "   - 增强：LOW -> MEDIUM -> HIGH 顺延一级。\n"
                    "   - 减弱：HIGH -> MEDIUM -> LOW 降退一级。\n"
                    "   - 注意：系统严格仅保留 LOW, MEDIUM, HIGH 三档，不允许设置为 MINIMAL。\n"
                    "4. 重要消歧提醒：\n"
                    "   - “低级模型”、“差一点的模型”、“高级模型”等是指变更 model（模型），绝对不要误当成 thinking_level（思考强度）！只有明确出现“思考”、“想一想”、“深度”时才变更 thinking_level。\n"
                    "5. 增减对话轮数：在当前上下文条数基础上加减相应条数（1轮=2条），范围 0~50。\n"
                    "6. 未提及或无需变更的字段，值必须输出为 null。\n\n"
                    "【输出 JSON 字段规范】\n"
                    "{\n"
                    '  "intent": "一句话客观描述你识别到的用户意图（例如：将模型由主力型切换至经济型）",\n'
                    '  "thinking_level": "LOW" | "MEDIUM" | "HIGH" | null,\n'
                    '  "context_messages": 整数(消息条数) | null,\n'
                    '  "search_enabled": true | false | null,\n'
                    '  "model": "gemini-3.5-flash-lite" | "gemini-3.8-flash" | "gemini-4-argon" | null,\n'
                    '  "max_output_tokens": 整数 | null\n'
                    "}\n\n"
                    f"用户指令: //{param_text}\n"
                    "请仅输出单个 JSON 对象："
                )
                try:
                    parse_resp = await self.client.generate_response(
                        messages=[{"role": "user", "content": extraction_prompt}],
                        model="gemini-3.5-flash-lite",
                        thinking_level="LOW",
                        enable_search=False,
                        max_output_tokens=256,
                        system_instruction="你是一个严格的参数提取器。只输出严格合法的单个JSON对象，不要输出任何多余问候、解释说明或Markdown代码块。",
                    )
                except Exception:
                    # Fallback to configured model if 3.5-flash-lite is unavailable
                    parse_resp = await self.client.generate_response(
                        messages=[{"role": "user", "content": extraction_prompt}],
                        model=self.settings.gemini_model,
                        thinking_level="LOW",
                        enable_search=False,
                        max_output_tokens=256,
                        system_instruction="你是一个严格的参数提取器。只输出严格合法的单个JSON对象，不要输出任何多余问候、解释说明或Markdown代码块。",
                    )

                raw_json = parse_resp.text.strip()
                json_match = re.search(r"\{.*?\}", raw_json, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    reply = await self._apply_extracted_settings(user_id, parsed)
                    if reply:
                        return reply

                return f"未能从指令 '//{param_text}' 中识别出有效配置变更。请输入 //help 查看指令列表与当前配置。"
            except Exception as e:
                logger.warning("[ChatManager] Fuzzy command extraction failed: %s", e, exc_info=True)
                return f"指令 '//{param_text}' 解析失败，请输入标准指令（如 //help 或 //低思考 或 //模型 3.8flash）。"

        # 2. Standard '/' commands
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/help", "/帮助", "/?"):
                return await self._get_help_text(user_id)

            elif cmd in ("/reset", "/重置"):
                count = await self.db.clear_history(user_id)
                await self.db.clear_session_memory(user_id)
                self._pending_images.pop(user_id, None)
                return f"【会话已重置】已清空你的历史对话记录（共清理 {count} 条消息）。"

            elif cmd in ("/status", "/状态"):
                uptime_sec = int(time.time() - self.start_time)
                hrs = uptime_sec // 3600
                mins = (uptime_sec % 3600) // 60
                secs = uptime_sec % 60
                user_cfg = await self._effective_user_settings(user_id)
                cur_model = user_cfg.get("model") or self.settings.gemini_model
                stats = await self.db.get_usage_summary(user_id)
                session_mem = await self.db.get_session_memory(user_id)
                summary_status = f"已生成 ({len(session_mem.get('summary', ''))}字)" if session_mem.get("summary") else "无 (未触发压缩)"
                pending_img_status = "有 (待发送提问)" if user_id in self._pending_images else "无"

                return (
                    "【智能助手运行状态】\n"
                    "----------------------\n"
                    f"- 运行时间: {hrs}小时{mins}分{secs}秒\n"
                    f"- 全局最大并发: {self.settings.max_gemini_concurrency}\n"
                    f"- 数据库状态: 正常 (WAL 模式)\n"
                    f"- 你的专属模型: {cur_model}\n"
                    f"- 会话记忆摘要: {summary_status}\n"
                    f"- 暂存待问图片: {pending_img_status}\n"
                    f"- 你的总请求数: {stats['user']['requests']} 次\n"
                    f"- 你的总消耗 Token: {stats['user']['total_tokens']}\n"
                    f"- 全局总请求数: {stats['global']['requests']} 次"
                )

            elif cmd in ("/model", "/模型"):
                if not arg:
                    user_cfg = await self._effective_user_settings(user_id)
                    cur_model = user_cfg.get("model") or self.settings.gemini_model
                    return (
                        f"【你的当前专属模型】: {cur_model}\n"
                        "切换命令示例: /model gemini-3.8-flash 或 /model gemini-3.5-flash-lite 或 /model gemini-4-argon\n"
                        "提示：系统已取消对 Gemini 2.5 旧模型的支持，目前仅支持 Gemini 3/4 新一代模型。"
                    )
                new_model = normalize_model_name(arg.strip())
                if "2.5" in new_model.lower():
                    return "【配置提示】系统已取消对 Gemini 2.5 旧模型的支持。请选择新一代模型：gemini-3.5-flash-lite、gemini-3.8-flash 或 gemini-4-argon。"
                await self.db.update_user_settings(user_id, model=new_model)
                return f"【专属模型已更新】你当前的独立模型已设置为: {new_model}"

            elif cmd in ("/search", "/搜索"):
                if arg.lower() in ("on", "1", "true", "开启", "开"):
                    await self.db.update_user_settings(user_id, search_enabled=True)
                    return "【实时搜索】已开启。遇到需要最新信息或专业领域的问题将自动执行 Google 搜索检索。"
                elif arg.lower() in ("off", "0", "false", "关闭", "关"):
                    await self.db.update_user_settings(user_id, search_enabled=False)
                    return "【实时搜索】已关闭。模型将仅依靠自带训练知识回答。"
                else:
                    return "【实时搜索指令】用法: /search on (开启) 或 /search off (关闭)"

            else:
                return f"未知指令 '{cmd}'，请输入 //help 查看可用指令列表。"

        return None

    async def _apply_command_jitter(self, user_id: str) -> float:
        """
        Apply 3.0 ~ 6.0 seconds random jitter delay to command responses.
        Simulates human reaction/typing speed and prevents robotic instant-reply risk detection.
        """
        min_jit = float(getattr(self.settings, "command_jitter_min_seconds", 3.0))
        max_jit = float(getattr(self.settings, "command_jitter_max_seconds", 6.0))
        if max_jit > 0 and max_jit >= min_jit:
            jitter = round(random.uniform(min_jit, max_jit), 2)
            logger.info(f"[ChatManager] Command anti-detection jitter: sleeping {jitter}s for user={user_id}")
            await asyncio.sleep(jitter)
            return jitter
        return 0.0

    async def process_message(
        self,
        user_id: str | int,
        content: str = "",
        message_id: Optional[str] = None,
        image_sources: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Main entry point for processing an incoming user message.
        Enforces whitelist, deduplication, FIFO per-user lock, concurrency limit,
        context assembly, multimodal image handling, and error masking.
        """
        uid = str(user_id).strip()
        text = (content or "").strip()

        if not text and not image_sources:
            return None

        # 1. Whitelist Verification
        if not self.settings.is_user_allowed(uid):
            logger.warning(f"[ChatManager] Rejected unauthorized message from QQ ID: {uid}")
            return "【提示】此 QQ 智能助手为私人服务，你的账号未在允许的白名单列表中。"

        # 2. Message Deduplication
        if message_id:
            mid = str(message_id)
            if mid in self._seen_msg_ids or await self.db.is_message_processed(mid):
                logger.info(f"[ChatManager] Ignored duplicate message_id: {mid}")
                return None
            self._seen_msg_ids.add(mid)
            if len(self._seen_msg_ids) > 1000:
                self._seen_msg_ids.pop()
            await self.db.mark_message_processed(mid)

        # 3. Check Commands (/help, //成本, etc.)
        if text:
            cmd_reply = await self.handle_command(uid, text)
            if cmd_reply is not None:
                await self._apply_command_jitter(uid)
                return cmd_reply

        # 4. Sequential Processing per user (FIFO Lock)
        user_lock = await self._get_user_lock(uid)
        async with user_lock:
            # Process any newly received images
            processed_images: List[Dict[str, Any]] = []
            if image_sources:
                for img_src in image_sources:
                    try:
                        raw_data = await download_image(img_src)
                        comp_bytes, mime, orig_size, final_size, strategy = compress_image(raw_data, max_long_edge=1200)
                        processed_images.append({
                            "bytes": comp_bytes,
                            "mime_type": mime,
                            "orig_size": orig_size,
                            "final_size": final_size,
                            "strategy": strategy,
                        })
                        logger.info(
                            f"[ChatManager] Image processed for user={uid}: strategy={strategy}, {orig_size} -> {final_size}, {len(comp_bytes)} bytes"
                        )
                    except Exception as e:
                        logger.error(f"[ChatManager] Failed to process image for user={uid}: {e}")
                        return f"【提示】图片下载或解析失败（{str(e)[:40]}），请重新发送。"

            # If user ONLY sent image(s) without text:
            if not text:
                if processed_images:
                    self._pending_images[uid] = {
                        "images": processed_images,
                        "timestamp": time.time(),
                    }
                    strategy_labels = {
                        "crop": "无损微切",
                        "bicubic": "双三次平滑",
                        "lanczos": "高解析度卷积",
                        "original": "原图直出",
                    }
                    dims_str = "、".join([
                        f"{img['final_size'][0]}x{img['final_size'][1]} [{strategy_labels.get(img.get('strategy', ''), '智能调校')}]"
                        for img in processed_images
                    ])
                    return f"已收到你的图片（已完成高解析度处理，尺寸: {dims_str}）。请发送你的问题或文字，我将结合图片为你解答。"
                return None

            # User sent text: determine which images to attach
            active_images: Optional[List[Dict[str, Any]]] = None
            if processed_images:
                # User sent image + text in the same message
                active_images = processed_images
                self._pending_images.pop(uid, None)
            elif uid in self._pending_images:
                # User previously staged image(s) and now sent follow-up text
                pending = self._pending_images.pop(uid)
                if (time.time() - pending["timestamp"]) <= 900:  # 15 minutes window
                    active_images = pending["images"]
                    logger.info(f"[ChatManager] Attached {len(active_images)} staged images to query for user={uid}")

            # Load user custom settings or fall back to defaults
            user_cfg = await self._effective_user_settings(uid)
            target_model = user_cfg.get("model") or self.settings.gemini_model
            thinking_lvl = user_cfg.get("thinking_level") or self.settings.default_thinking_level
            ctx_limit = user_cfg["context_messages"]
            # Search intent distinguishes an explicit opt-out from an inherited default.
            user_search = (await self.db.get_user_settings(uid) or {}).get("search_enabled")
            target_max_tokens = user_cfg.get("max_output_tokens") or self.settings.max_output_tokens

            # Determine whether search is needed
            enable_search = self._should_trigger_search(text, user_search)

            # Session memory & dynamic context window management
            now = time.time()
            session_mem = await self.db.get_session_memory(uid)
            last_active = session_mem.get("last_active_at", 0.0)
            is_continuous = (now - last_active) <= self.settings.session_timeout_seconds

            messages: List[Dict[str, str]] = []
            active_summary = session_mem.get("summary", "")

            if is_continuous and last_active > 0:
                # Active continuous session: retrieve uncompressed messages since compressed_msg_id
                compressed_id = session_mem.get("compressed_msg_id", 0)
                uncompressed = await self.db.get_uncompressed_messages(uid, compressed_id)

                if active_summary:
                    messages.append({
                        "role": "user",
                        "content": f"【前文对话背景摘要】\n{active_summary}\n（这是此前连续讨论的核心脉络，请结合该背景与下文最新对话进行回答。）"
                    })
                    messages.append({
                        "role": "model",
                        "content": "收到，我已掌握前文的核心讨论背景与事实脉络，将结合这一上下文继续与你沟通。"
                    })

                for m in uncompressed:
                    messages.append({"role": m["role"], "content": m["content"]})
            else:
                # Inactive / new session break: clear prior session summary and use standard recent context
                if last_active > 0 and not is_continuous:
                    await self.db.clear_session_memory(uid)
                    session_mem = await self.db.get_session_memory(uid)
                history = await self.db.get_recent_messages(uid, limit=ctx_limit)
                for m in history:
                    messages.append({"role": m["role"], "content": m["content"]})

            messages.append({"role": "user", "content": text})

            # 5. Global Concurrency Limit (Semaphore)
            try:
                async with self.concurrency_semaphore:
                    logger.info(
                        f"[ChatManager] Calling Gemini API for user={uid}, model={target_model}, "
                        f"ctx={len(messages)-1}, search={enable_search}, thinking={thinking_lvl}, "
                        f"images={len(active_images) if active_images else 0}, tokens={target_max_tokens}"
                    )
                    response = await asyncio.wait_for(
                        self.client.generate_response(
                            messages=messages,
                            model=target_model,
                            thinking_level=thinking_lvl,
                            enable_search=enable_search,
                            max_output_tokens=target_max_tokens,
                            images=active_images,
                        ),
                        timeout=self.settings.user_request_timeout,
                    )
            except asyncio.TimeoutError:
                logger.error(f"[ChatManager] Request timed out for user={uid}")
                return "【提示】请求超时，AI 思考或网络连接耗时过长，请稍后再试。"
            except Exception as e:
                logger.error(f"[ChatManager] Gemini API call error for user={uid}: {e}", exc_info=False)
                # Map error to safe friendly messages without leaking internals
                err_str = str(e)
                code = getattr(e, "code", None)
                if code == 429 or "429" in err_str or "ResourceExhausted" in err_str:
                    return "【提示】当前请求过于频繁或触发了 API 频率限制，请等待 10-20 秒后重试。"
                elif code in (401, 403) or "403" in err_str or "API_KEY_INVALID" in err_str or "401" in err_str:
                    return "【提示】服务配置或授权异常，请联系管理员检查 API 密钥设置。"
                return "【提示】暂时无法连接至 AI 服务，请稍后重新发送。"

            # 6. Save message history to SQLite
            saved_user_msg = f"[图片] {text}" if active_images else text
            await self.db.save_message(uid, "user", saved_user_msg)
            await self.db.save_message(uid, "model", response.text)

            # Update session last active time
            await self.db.update_session_memory(uid, last_active_at=now)

            # Check if active uncompressed messages reached compression threshold
            current_compressed_id = session_mem.get("compressed_msg_id", 0)
            uncompressed_after = await self.db.get_uncompressed_messages(uid, current_compressed_id)
            threshold_msgs = self.settings.session_compression_threshold_rounds * 2
            if len(uncompressed_after) >= threshold_msgs:
                asyncio.create_task(self.compress_session_history(uid))

            # 7. Record token usage and search metrics
            await self.db.record_usage(
                user_id=uid,
                prompt_tokens=response.prompt_tokens,
                candidate_tokens=response.candidate_tokens,
                search_queries=len(response.search_queries),
                model_used=response.model_used or target_model,
            )

            # 8. Format final reply for QQ
            return self._format_reply(response, text)
