"""
Comprehensive Automated Test Suite for QQ Gemini Assistant.
Verifies:
1. Multi-user session isolation
2. Message processing & context continuity
3. /reset command functionality
4. Timeout and exception handling (429, timeouts, auth error)
5. Message deduplication (in-memory & SQLite)
6. Whitelist access control
7. Database persistence across restarts
8. API key masking and protection
9. Fuzzy & standard commands (//成本, //低思考, //5轮对话记录)
10. Search Grounding triggering logic
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import pytest
import pytest_asyncio

from config import Settings
from db import Database
from mock_gemini import MockGeminiClient
from chat_manager import ChatManager
from qq_adapter.onebot_v11 import OneBotV11Adapter


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="qq_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_settings(temp_dir):
    db_file = os.path.join(temp_dir, "test.db")
    return Settings(
        gemini_api_key="TEST_MOCK_KEY_SECRET",
        gemini_model="gemini-3.8-flash",
        default_thinking_level="MEDIUM",
        default_context_messages=4,
        max_output_tokens=1024,
        enable_search_default=False,
        gemini_timeout_seconds=2.0,
        gemini_max_retries=2,
        qq_whitelist_raw="10001,10002,10003",
        max_gemini_concurrency=2,
        user_request_timeout=5.0,
        sqlite_db_path=db_file,
        command_jitter_min_seconds=0.0,
        command_jitter_max_seconds=0.0,
    )


@pytest_asyncio.fixture
async def assistant_env(test_settings):
    db = Database(test_settings.sqlite_db_path)
    await db.init_db()
    mock_client = MockGeminiClient(default_model=test_settings.gemini_model, simulate_delay=0.01)
    chat_mgr = ChatManager(settings=test_settings, db=db, client=mock_client)
    return test_settings, db, mock_client, chat_mgr


@pytest.mark.asyncio
async def test_whitelist_access_control(assistant_env):
    """Verify whitelist rejects strangers and permits authorized users."""
    settings, db, mock_client, chat_mgr = assistant_env

    # 1. Authorized user
    reply_auth = await chat_mgr.process_message(user_id="10001", content="你好")
    assert reply_auth is not None
    assert "Gemini" in reply_auth or "你好" in reply_auth

    # 2. Unauthorized stranger
    reply_unauth = await chat_mgr.process_message(user_id="99999", content="你好")
    assert reply_unauth is not None
    assert "未在允许的白名单" in reply_unauth
    assert mock_client.call_count == 1  # Client should not have been called for unauthorized user!


@pytest.mark.asyncio
async def test_multi_user_isolation(assistant_env):
    """Verify that different users have completely isolated message histories."""
    settings, db, mock_client, chat_mgr = assistant_env

    # User 1 sends a secret topic
    await chat_mgr.process_message(user_id="10001", content="我的秘密水果是蓝莓")

    # User 2 sends a message
    await chat_mgr.process_message(user_id="10002", content="我喜欢吃苹果")

    # User 1 sends follow-up
    await chat_mgr.process_message(user_id="10001", content="请问我刚才提到的水果是什么？")
    user1_history = await db.get_recent_messages("10001", limit=10)
    user2_history = await db.get_recent_messages("10002", limit=10)

    # Check isolation in DB
    user1_texts = [m["content"] for m in user1_history]
    user2_texts = [m["content"] for m in user2_history]

    assert any("蓝莓" in t for t in user1_texts)
    assert not any("苹果" in t for t in user1_texts)

    assert any("苹果" in t for t in user2_texts)
    assert not any("蓝莓" in t for t in user2_texts)


@pytest.mark.asyncio
async def test_concurrent_multi_user_no_crosstalk(assistant_env):
    """Verify that multiple users chatting concurrently never cross talk or leak data."""
    settings, db, mock_client, chat_mgr = assistant_env
    mock_client.simulate_delay = 0.05  # Ensure asynchronous interleaving

    # Both users initiate conversations simultaneously
    res1, res2 = await asyncio.gather(
        chat_mgr.process_message(user_id="10001", content="秘密A_10001"),
        chat_mgr.process_message(user_id="10002", content="秘密B_10002"),
    )
    assert res1 is not None
    assert res2 is not None

    # Follow-up queries concurrently
    res1_followup, res2_followup = await asyncio.gather(
        chat_mgr.process_message(user_id="10001", content="我刚才说的密码是什么？"),
        chat_mgr.process_message(user_id="10002", content="我刚才说的密码是什么？"),
    )

    # Verify database isolation
    history1 = await db.get_recent_messages("10001", limit=10)
    history2 = await db.get_recent_messages("10002", limit=10)

    texts1 = [m["content"] for m in history1]
    texts2 = [m["content"] for m in history2]

    assert any("秘密A_10001" in t for t in texts1)
    assert not any("秘密B_10002" in t for t in texts1)

    assert any("秘密B_10002" in t for t in texts2)
    assert not any("秘密A_10001" in t for t in texts2)

    # Same user sends two messages rapidly (FIFO serialized per user)
    r1, r2 = await asyncio.gather(
        chat_mgr.process_message(user_id="10001", content="快速并发消息1"),
        chat_mgr.process_message(user_id="10001", content="快速并发消息2"),
    )
    assert r1 is not None and r2 is not None


@pytest.mark.asyncio
async def test_context_continuity_and_limit(assistant_env):
    """Verify multi-turn history continuity in continuous session and fallback to static limit after timeout."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # Send 3 rounds (6 messages)
    await chat_mgr.process_message(user_id=user_id, content="第一问")
    await chat_mgr.process_message(user_id=user_id, content="第二问")
    await chat_mgr.process_message(user_id=user_id, content="第三问")

    # In continuous active session (within 10 minutes), all 6 historical messages are retained
    await chat_mgr.process_message(user_id=user_id, content="第四问")
    passed_continuous = mock_client.last_messages
    assert len(passed_continuous) == 7  # 6 historical + 1 current
    assert passed_continuous[-1]["content"] == "第四问"

    # Simulate session timeout (inactive for > 600s)
    await db.update_session_memory(user_id, last_active_at=1.0)

    # After session timeout, falls back to static limit (default_context_messages = 4)
    await chat_mgr.process_message(user_id=user_id, content="第五问")
    passed_fallback = mock_client.last_messages
    assert len(passed_fallback) == 5  # 4 historical + 1 current
    assert passed_fallback[-1]["content"] == "第五问"


@pytest.mark.asyncio
async def test_reset_command(assistant_env):
    """Verify /reset clears history for the user and keeps other users unaffected."""
    settings, db, mock_client, chat_mgr = assistant_env

    await chat_mgr.process_message(user_id="10001", content="测试消息1")
    await chat_mgr.process_message(user_id="10002", content="用户2的消息")

    # Reset user 1
    reply = await chat_mgr.process_message(user_id="10001", content="/reset")
    assert "已清空你的历史对话记录" in reply

    # User 1 history should now be empty
    u1_msgs = await db.get_recent_messages("10001", limit=10)
    assert len(u1_msgs) == 0

    # User 2 history should remain intact
    u2_msgs = await db.get_recent_messages("10002", limit=10)
    assert len(u2_msgs) > 0


@pytest.mark.asyncio
async def test_message_deduplication(assistant_env):
    """Verify duplicate message_id is processed only once."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"
    msg_id = "unique_qq_msg_12345"

    reply1 = await chat_mgr.process_message(user_id=user_id, content="测试消息", message_id=msg_id)
    assert reply1 is not None
    initial_calls = mock_client.call_count

    # Second arrival of the same message_id
    reply2 = await chat_mgr.process_message(user_id=user_id, content="测试消息", message_id=msg_id)
    assert reply2 is None  # Silently ignored
    assert mock_client.call_count == initial_calls


@pytest.mark.asyncio
async def test_timeout_and_error_masking(assistant_env):
    """Verify timeouts and errors are handled and masked without leaking internals."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # 1. Timeout simulation
    mock_client.fail_with_timeout_count = 1
    reply = await chat_mgr.process_message(user_id=user_id, content="测试超时")
    assert "【提示】" in reply
    assert "Traceback" not in reply
    assert "TEST_MOCK_KEY_SECRET" not in reply

    # 2. 429 Rate limit simulation
    mock_client.fail_with_429_count = 1
    reply_429 = await chat_mgr.process_message(user_id=user_id, content="测试频率限制")
    assert "频率限制" in reply_429 or "稍后" in reply_429
    assert "ResourceExhausted" not in reply_429

    # 3. Auth error simulation
    mock_client.fail_with_auth_error = True
    reply_auth = await chat_mgr.process_message(user_id=user_id, content="测试密钥异常")
    assert "授权" in reply_auth or "管理员" in reply_auth
    assert "TEST_MOCK_KEY_SECRET" not in reply_auth


@pytest.mark.asyncio
async def test_database_persistence_restart(test_settings):
    """Verify that restarting the service preserves SQLite chat history and settings."""
    db_path = test_settings.sqlite_db_path

    # First run
    db1 = Database(db_path)
    await db1.init_db()
    client1 = MockGeminiClient()
    mgr1 = ChatManager(test_settings, db1, client1)

    await mgr1.process_message(user_id="10001", content="持久化测试消息")
    await mgr1.process_message(user_id="10001", content="//高思考")

    # Simulate shutdown and reboot with new instances
    del mgr1
    del db1

    db2 = Database(db_path)
    await db2.init_db()
    client2 = MockGeminiClient()
    mgr2 = ChatManager(test_settings, db2, client2)

    # Check restored messages
    history = await db2.get_recent_messages("10001", limit=10)
    assert len(history) >= 2  # user + model
    assert history[0]["content"] == "持久化测试消息"

    # Check restored user settings
    saved_cfg = await db2.get_user_settings("10001")
    assert saved_cfg["thinking_level"] == "HIGH"


@pytest.mark.asyncio
async def test_commands_and_cost_accounting(assistant_env):
    """Verify /help, //help, /status, /model, //成本, //低思考, //3轮对话记录."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # 1. /help and //help
    help_resp = await chat_mgr.process_message(user_id, "/help")
    assert "指令列表" in help_resp or "帮助" in help_resp

    help_resp_slash = await chat_mgr.process_message(user_id, "//help")
    assert "指令列表" in help_resp_slash or "帮助" in help_resp_slash

    # 2. //低思考
    think_resp = await chat_mgr.process_message(user_id, "//低思考")
    assert "LOW" in think_resp
    cfg = await db.get_user_settings(user_id)
    assert cfg["thinking_level"] == "LOW"

    # 3. //3轮对话记录
    round_resp = await chat_mgr.process_message(user_id, "//3轮对话记录")
    assert "3 轮对话" in round_resp
    cfg = await db.get_user_settings(user_id)
    assert cfg["context_messages"] == 6

    # 4. Generate some usage
    await chat_mgr.process_message(user_id, "普通问答测试")

    # 5. //成本
    cost_resp = await chat_mgr.process_message(user_id, "//成本")
    assert "成本与 Token 统计" in cost_resp
    assert "个人累计用量" in cost_resp
    assert "Token:" in cost_resp


@pytest.mark.asyncio
async def test_per_user_model_isolation(assistant_env):
    """Verify that each user can configure and maintain a distinct Gemini model."""
    settings, db, mock_client, chat_mgr = assistant_env

    # User 10001 configures gemini-3.8-flash
    resp1 = await chat_mgr.process_message("10001", "//模型 gemini-3.8-flash")
    assert "gemini-3.8-flash" in resp1

    # User 10002 configures gemini-3.5-flash-lite
    resp2 = await chat_mgr.process_message("10002", "//模型 gemini-3.5-flash-lite")
    assert "gemini-3.5-flash-lite" in resp2

    # User 10001 sends a chat message -> client receives gemini-3.8-flash
    await chat_mgr.process_message("10001", "用户10001问答")
    cfg1 = await db.get_user_settings("10001")
    assert cfg1["model"] == "gemini-3.8-flash"

    # User 10002 sends a chat message -> client receives gemini-3.5-flash-lite
    await chat_mgr.process_message("10002", "用户10002问答")
    cfg2 = await db.get_user_settings("10002")
    assert cfg2["model"] == "gemini-3.5-flash-lite"

    # Query individual models via //模型 without arguments
    check1 = await chat_mgr.process_message("10001", "//模型")
    assert "gemini-3.8-flash" in check1

    check2 = await chat_mgr.process_message("10002", "//模型")
    assert "gemini-3.5-flash-lite" in check2


@pytest.mark.asyncio
async def test_markdown_and_links_suppression(assistant_env):
    """Verify that Markdown formatting and unrequested links are suppressed."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # Enable search
    await chat_mgr.process_message(user_id, "//搜索 开")

    # 1. Query without requesting links: should NOT contain citations
    reply_normal = await chat_mgr.process_message(user_id, "头孢用药说明")
    assert "参考来源" not in reply_normal
    assert "https://" not in reply_normal
    # Should not contain markdown bold **
    assert "**" not in reply_normal

    # 2. Query explicitly requesting links: SHOULD contain citations
    reply_with_links = await chat_mgr.process_message(user_id, "头孢用药说明，请提供参考来源链接")
    assert "参考来源" in reply_with_links
    assert "https://" in reply_with_links


@pytest.mark.asyncio
async def test_search_intent_detection(assistant_env):
    """Verify Search Grounding is triggered for disease, hardware, and 2025/2026 queries."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # Medical question -> must search
    await chat_mgr.process_message(user_id, "头孢和阿莫西林有什么用药区别？")
    assert mock_client.last_enable_search is True

    # Hardware question -> must search
    await chat_mgr.process_message(user_id, "RTX 5090 显卡功耗是多少？")
    assert mock_client.last_enable_search is True

    # Year 2026 question -> must search
    await chat_mgr.process_message(user_id, "2026年澳大利亚假期安排")
    assert mock_client.last_enable_search is True

    # Casual greeting without mandatory keywords and search default disabled -> should NOT search
    await chat_mgr.process_message(user_id, "你好，很高兴认识你")
    assert mock_client.last_enable_search is False


@pytest.mark.asyncio
async def test_fuzzy_commands_and_aliases(assistant_env):
    """
    Verify robust fuzzy commands, alias normalization, and parameter parsing:
    - //设置为10轮对话记录
    - //把思考调成低
    - //设置模型为3.8flash
    - //切换模型为3.5lite
    - //开启搜索
    - //输出长度 4096
    - Multi-property commands
    - Fallback to AI parsing
    """
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # 1. //设置为10轮对话记录
    resp1 = await chat_mgr.process_message(user_id, "//设置为10轮对话记录")
    assert "智能配置已生效" in resp1
    assert "20 条消息" in resp1
    cfg1 = await db.get_user_settings(user_id)
    assert cfg1["context_messages"] == 20

    # 2. //把思考调成低
    resp2 = await chat_mgr.process_message(user_id, "//把思考调成低")
    assert "智能配置已生效" in resp2
    assert "思考强度=LOW" in resp2
    cfg2 = await db.get_user_settings(user_id)
    assert cfg2["thinking_level"] == "LOW"

    # 3. //设置模型为3.8flash (model alias normalization)
    resp3 = await chat_mgr.process_message(user_id, "//设置模型为3.8flash")
    assert "智能配置已生效" in resp3
    assert "专属模型=gemini-3.8-flash" in resp3
    cfg3 = await db.get_user_settings(user_id)
    assert cfg3["model"] == "gemini-3.8-flash"

    # 4. //切换模型为3.5lite
    resp4 = await chat_mgr.process_message(user_id, "//切换模型为3.5lite")
    assert "智能配置已生效" in resp4
    assert "专属模型=gemini-3.5-flash-lite" in resp4
    cfg4 = await db.get_user_settings(user_id)
    assert cfg4["model"] == "gemini-3.5-flash-lite"

    # 5. //开启搜索
    resp5 = await chat_mgr.process_message(user_id, "//开启搜索")
    assert "智能配置已生效" in resp5 or "实时搜索" in resp5
    cfg5 = await db.get_user_settings(user_id)
    assert cfg5["search_enabled"] is True

    # 6. //输出长度 4096
    resp6 = await chat_mgr.process_message(user_id, "//输出长度 4096")
    assert "智能配置已生效" in resp6
    assert "最大输出=4096 tokens" in resp6
    cfg6 = await db.get_user_settings(user_id)
    assert cfg6["max_output_tokens"] == 4096

    # 7. Multi-property combined command
    resp7 = await chat_mgr.process_message(user_id, "//把思考调成中，保留5轮对话记录")
    assert "智能配置已生效" in resp7
    assert "思考强度=MEDIUM" in resp7
    assert "10 条消息" in resp7
    cfg7 = await db.get_user_settings(user_id)
    assert cfg7["thinking_level"] == "MEDIUM"
    assert cfg7["context_messages"] == 10

    # 8. Natural language with keywords
    resp8 = await chat_mgr.process_message(user_id, "//麻烦帮我把思考设为高思考并记录8条历史")
    assert "智能配置已生效" in resp8
    cfg8 = await db.get_user_settings(user_id)
    assert cfg8["thinking_level"] == "HIGH"
    assert cfg8["context_messages"] == 8

    # 9. Natural language without digits (forces AI fallback extraction)
    resp9 = await chat_mgr.process_message(user_id, "//麻烦用最便宜的3.5模型，上下文保留十轮")
    assert "智能配置已生效" in resp9
    assert "专属模型=gemini-3.5-flash-lite" in resp9
    assert "20 条消息" in resp9
    cfg9 = await db.get_user_settings(user_id)
    assert cfg9["model"] == "gemini-3.5-flash-lite"
    assert cfg9["context_messages"] == 20

    # 10. Relative thinking adjustment: 思考强度更高一点 (LOW -> MEDIUM)
    await chat_mgr.process_message(user_id, "//低思考")
    resp10 = await chat_mgr.process_message(user_id, "//思考强度更高一点")
    assert "智能配置已生效" in resp10
    assert "思考强度=MEDIUM" in resp10
    cfg10 = await db.get_user_settings(user_id)
    assert cfg10["thinking_level"] == "MEDIUM"

    # 11. Relative model upgrade: 帮我换个更好的模型 (3.5 -> 3.8 -> 4-argon)
    await chat_mgr.process_message(user_id, "//模型 gemini-3.5-flash-lite")
    resp11_a = await chat_mgr.process_message(user_id, "//帮我换个更好的模型")
    assert "专属模型=gemini-3.8-flash" in resp11_a
    cfg11_a = await db.get_user_settings(user_id)
    assert cfg11_a["model"] == "gemini-3.8-flash"

    resp11_b = await chat_mgr.process_message(user_id, "//帮我换个更好的模型")
    assert "专属模型=gemini-4-argon" in resp11_b
    cfg11_b = await db.get_user_settings(user_id)
    assert cfg11_b["model"] == "gemini-4-argon"

    # 12. Relative model downgrade: 换成最便宜的模型
    resp12 = await chat_mgr.process_message(user_id, "//换成最便宜的模型")
    assert "专属模型=gemini-3.5-flash-lite" in resp12
    cfg12 = await db.get_user_settings(user_id)
    assert cfg12["model"] == "gemini-3.5-flash-lite"

    # 13. Relative model downgrade step-by-step: 切换到低级一点的模型 (4-argon -> 3.8-flash -> 3.5-flash-lite)
    await chat_mgr.process_message(user_id, "//模型 gemini-4-argon")
    resp13_a = await chat_mgr.process_message(user_id, "//切换到低级一点的模型")
    assert "专属模型=gemini-3.8-flash" in resp13_a
    cfg13_a = await db.get_user_settings(user_id)
    assert cfg13_a["model"] == "gemini-3.8-flash"

    resp13_b = await chat_mgr.process_message(user_id, "//切换到低级一点的模型")
    assert "专属模型=gemini-3.5-flash-lite" in resp13_b
    cfg13_b = await db.get_user_settings(user_id)
    assert cfg13_b["model"] == "gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_session_compression_and_summary_buffer(assistant_env):
    """
    Verify the 10-round session compression mechanism:
    - 1~10 rounds in continuous chat are kept verbatim.
    - Exceeding 10 rounds triggers gemini-3.5-flash-lite to compress rounds older than recent 4 rounds.
    - The summary is prepended to subsequent turns alongside recent 4 rounds verbatim.
    - Summary persists on subsequent turns.
    - /reset completely wipes the session summary and history.
    """
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # Send 10 rounds of chat (20 messages: 10 user + 10 model)
    for i in range(1, 11):
        await chat_mgr.process_message(user_id, f"连续会话第{i}问")

    # The 10th round triggers compress_session_history in the background.
    # Yield control briefly to let the background compression task finish.
    await asyncio.sleep(0.1)

    # Verify session_memory state in DB
    session_mem = await db.get_session_memory(user_id)
    assert session_mem["summary"] != ""
    assert "前文背景摘要" in session_mem["summary"]
    assert session_mem["compressed_msg_id"] > 0

    # Round 11: should transmit [Summary user+model] + [Recent 4 rounds uncompressed (8 msgs)] + [Current question]
    # Total messages to client = 2 (summary turn) + 8 (uncompressed messages) + 1 (current) = 11 messages!
    await chat_mgr.process_message(user_id, "连续会话第11问")
    msgs_11 = mock_client.last_messages
    assert len(msgs_11) == 11
    assert "前文对话背景摘要" in msgs_11[0]["content"]
    assert msgs_11[0]["role"] == "user"
    assert msgs_11[1]["role"] == "model"
    assert msgs_11[-1]["content"] == "连续会话第11问"

    # Round 12: summary continues to persist, now with 10 uncompressed messages + 1 current = 13 messages
    await chat_mgr.process_message(user_id, "连续会话第12问")
    msgs_12 = mock_client.last_messages
    assert len(msgs_12) == 13
    assert "前文对话背景摘要" in msgs_12[0]["content"]
    assert msgs_12[-1]["content"] == "连续会话第12问"

    # Reset clears both history and session memory
    await chat_mgr.process_message(user_id, "/reset")
    cleared_mem = await db.get_session_memory(user_id)
    assert cleared_mem["summary"] == ""
    assert cleared_mem["compressed_msg_id"] == 0


def create_test_image_b64(width: int, height: int, color: tuple = (255, 0, 0)) -> str:
    im = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    raw = buf.getvalue()
    return "base64://" + base64.b64encode(raw).decode("ascii")


@pytest.mark.asyncio
async def test_image_staging_and_followup_text(assistant_env):
    """
    Verify:
    1. User sends an image alone (> 1200px long edge).
    2. Image is compressed to max 1200px and staged in memory.
    3. User sends follow-up text, and the staged image is delivered to Gemini with the text.
    4. Database logs '[图片] <text>'.
    5. Staging buffer is consumed and cleared.
    """
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # User sends a 2400x1600 image alone
    img_b64 = create_test_image_b64(2400, 1600, color=(10, 80, 200))
    reply1 = await chat_mgr.process_message(
        user_id=user_id,
        content="",
        image_sources=[img_b64],
    )

    assert reply1 is not None
    assert "已收到你的图片" in reply1
    assert "1200x800" in reply1  # 2400x1600 scaled down by factor of 2 to 1200px long edge!
    assert user_id in chat_mgr._pending_images

    # User sends follow-up text
    reply2 = await chat_mgr.process_message(
        user_id=user_id,
        content="请问图里的物体是什么颜色？",
    )

    assert reply2 is not None
    assert "已识别你发送的 1 张图片" in reply2
    assert user_id not in chat_mgr._pending_images  # Staged image consumed!
    assert len(mock_client.last_images) == 1
    assert mock_client.last_images[0]["final_size"] == (1200, 800)

    # Check database history
    history = await db.get_recent_messages(user_id, limit=5)
    assert any("[图片] 请问图里的物体是什么颜色？" in m["content"] for m in history)


@pytest.mark.asyncio
async def test_image_with_text_combined(assistant_env):
    """Verify user sending image + text in a single message."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # 800x600 image (long edge <= 1200, no downscaling needed)
    img_b64 = create_test_image_b64(800, 600, color=(0, 200, 100))
    reply = await chat_mgr.process_message(
        user_id=user_id,
        content="请直接帮我分析这张图片",
        image_sources=[img_b64],
    )

    assert reply is not None
    assert "已识别你发送的 1 张图片" in reply
    assert len(mock_client.last_images) == 1
    assert mock_client.last_images[0]["final_size"] == (800, 600)
    assert user_id not in chat_mgr._pending_images


@pytest.mark.asyncio
async def test_image_reset_clears_staging(assistant_env):
    """Verify /reset clears pending staged images."""
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    img_b64 = create_test_image_b64(1000, 1000)
    await chat_mgr.process_message(user_id=user_id, content="", image_sources=[img_b64])
    assert user_id in chat_mgr._pending_images

    await chat_mgr.process_message(user_id=user_id, content="/reset")
    assert user_id not in chat_mgr._pending_images


@pytest.mark.asyncio
async def test_image_3_tier_adaptive_switching(assistant_env):
    """
    Verify the 3-tier adaptive switching:
    1. diff <= 100px (e.g. 1280x800) -> direct center crop (1200x800, strategy='crop')
    2. diff > 100px and long_edge <= 1600 (e.g. 1500x1000) -> Bicubic interpolation (1200x800, strategy='bicubic')
    3. long_edge > 1600 (e.g. 2400x1600) -> Lanczos convolution (1200x800, strategy='lanczos')
    4. long_edge <= 1200 (e.g. 900x600) -> Original untouched (900x600, strategy='original')
    """
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10001"

    # Tier 1: 1280x800 -> Diff = 80 <= 100 -> Crop
    b64_crop = create_test_image_b64(1280, 800)
    rep_crop = await chat_mgr.process_message(user_id, "", image_sources=[b64_crop])
    assert "无损微切" in rep_crop
    assert "1200x800" in rep_crop
    assert chat_mgr._pending_images[user_id]["images"][0]["strategy"] == "crop"
    chat_mgr._pending_images.clear()

    # Tier 2: 1500x1000 -> Diff = 300 > 100, <= 1600 -> Bicubic
    b64_bicubic = create_test_image_b64(1500, 1000)
    rep_bicubic = await chat_mgr.process_message(user_id, "", image_sources=[b64_bicubic])
    assert "双三次平滑" in rep_bicubic
    assert "1200x800" in rep_bicubic
    assert chat_mgr._pending_images[user_id]["images"][0]["strategy"] == "bicubic"
    chat_mgr._pending_images.clear()

    # Tier 3: 2400x1600 -> Long edge > 1600 -> Lanczos
    b64_lanczos = create_test_image_b64(2400, 1600)
    rep_lanczos = await chat_mgr.process_message(user_id, "", image_sources=[b64_lanczos])
    assert "高解析度卷积" in rep_lanczos
    assert "1200x800" in rep_lanczos
    assert chat_mgr._pending_images[user_id]["images"][0]["strategy"] == "lanczos"
    chat_mgr._pending_images.clear()

    # Tier 0: 900x600 -> Long edge <= 1200 -> Original
    b64_orig = create_test_image_b64(900, 600)
    rep_orig = await chat_mgr.process_message(user_id, "", image_sources=[b64_orig])
    assert "原图直出" in rep_orig
    assert "900x600" in rep_orig
    assert chat_mgr._pending_images[user_id]["images"][0]["strategy"] == "original"
    chat_mgr._pending_images.clear()


@pytest.mark.asyncio
async def test_no_minimal_thinking_and_gemini_2_5_rejection(assistant_env):
    """
    Verify:
    1. MINIMAL thinking level is completely disallowed and clamped to LOW across all components.
    2. Only 3 thinking levels (LOW, MEDIUM, HIGH) exist.
    3. Gemini 2.5 models are deprecated and rejected with friendly guidance.
    4. GoogleGenAIClient throws ValueError if 2.5 model is invoked.
    """
    settings, db, mock_client, chat_mgr = assistant_env
    user_id = "10002"

    # 1. Direct command requesting MINIMAL / 最小思考 / 无思考 -> clamped to LOW
    resp1 = await chat_mgr.process_message(user_id, "//把思考设为最小")
    assert "智能配置已生效" in resp1
    assert "思考强度=LOW" in resp1
    cfg1 = await db.get_user_settings(user_id)
    assert cfg1["thinking_level"] == "LOW"

    resp2 = await chat_mgr.process_message(user_id, "//关闭思考")
    assert "思考强度=LOW" in resp2
    cfg2 = await db.get_user_settings(user_id)
    assert cfg2["thinking_level"] == "LOW"

    # Relative decrease from LOW stays at LOW
    resp3 = await chat_mgr.process_message(user_id, "//思考再低一点")
    assert "思考强度=LOW" in resp3
    cfg3 = await db.get_user_settings(user_id)
    assert cfg3["thinking_level"] == "LOW"

    # 2. Reject Gemini 2.5 via /model command
    resp_m1 = await chat_mgr.process_message(user_id, "/model gemini-2.5-flash")
    assert "取消对 Gemini 2.5 旧模型的支持" in resp_m1
    cfg_m1 = await db.get_user_settings(user_id)
    assert not cfg_m1.get("model") or "2.5" not in cfg_m1["model"]

    # Reject Gemini 2.5 via fuzzy command
    resp_m2 = await chat_mgr.process_message(user_id, "//切换模型为2.5")
    assert "取消对 Gemini 2.5 旧模型的支持" in resp_m2

    # 3. Verify GeminiClient _map_thinking_level and generate_response 2.5 rejection
    from gemini_client import GoogleGenAIClient
    from google.genai import types

    real_client = GoogleGenAIClient(api_key="fake_key_for_unit_test")
    # MINIMAL and NONE are mapped to LOW
    assert real_client._map_thinking_level("MINIMAL") == types.ThinkingLevel.LOW
    assert real_client._map_thinking_level("minimal") == types.ThinkingLevel.LOW
    assert real_client._map_thinking_level("NONE") == types.ThinkingLevel.LOW
    assert real_client._map_thinking_level("LOW") == types.ThinkingLevel.LOW
    assert real_client._map_thinking_level("MEDIUM") == types.ThinkingLevel.MEDIUM
    assert real_client._map_thinking_level("HIGH") == types.ThinkingLevel.HIGH

    # Calling generate_response with 2.5 model raises ValueError
    import pytest
    with pytest.raises(ValueError, match="Gemini 2.5 系列模型.*已被淘汰并停止支持"):
        await real_client.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash"
        )





@pytest.mark.asyncio
async def test_nullable_settings_fuzzy_extraction(assistant_env, monkeypatch):
    """A model-only override leaves other SQLite columns NULL, not defaults."""
    from gemini_client import GeminiResponse
    settings, db, client, manager = assistant_env
    await db.update_user_settings("10001", model="gemini-4-argon")
    raw = await db.get_user_settings("10001")
    assert raw["context_messages"] is None
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs)
        return GeminiResponse(text='{"thinking_level":"HIGH","context_messages":null}')

    monkeypatch.setattr(client, "generate_response", extract)
    reply = await manager.handle_command("10001", "//请多动动脑再回答")
    assert "HIGH" in reply
    assert len(calls) == 1
    prompt = calls[0]["messages"][0]["content"]
    assert "4 条消息" in prompt
    help_text = await manager.handle_command("10001", "//help")
    assert "4 条消息" in help_text


@pytest.mark.asyncio
async def test_zero_context_and_false_search_survive_defaults(assistant_env):
    settings, db, client, manager = assistant_env
    settings.enable_search_default = True
    await db.update_user_settings("10001", context_messages=0, search_enabled=False)
    effective = await manager._effective_user_settings("10001")
    assert effective["context_messages"] == 0
    assert effective["search_enabled"] is False
    assert "0 条消息" in await manager.handle_command("10001", "//help")
    assert "关闭" in await manager.handle_command("10001", "//搜索")
    await manager.handle_command("10001", "//多记2轮对话")
    assert (await db.get_user_settings("10001"))["context_messages"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("model,expected", [
    ("gemini-4-argon", "gemini-3.8-flash"),
    ("gemini-3.8-flash", "gemini-3.5-flash-lite"),
    ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
])
async def test_colloquial_model_downgrade_offline(assistant_env, model, expected):
    settings, db, client, manager = assistant_env
    await db.update_user_settings("10001", model=model)
    reply = await manager.handle_command("10001", "//换个笨一点的模型")
    assert expected in reply
    assert client.call_count == 0
    assert (await db.get_user_settings("10001"))["model"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("command,heading", [
    ("//help 模型", "【2. 模型"),
    ("//帮助 图片", "【7. 图片"),
    ("/help thinking", "【3. 思考"),
    ("/? 记忆", "【5. 记忆"),
    ("//HELP 输出", "【6. 输出"),
    ("//help 配置", "【你当前的配置】"),
])
async def test_topic_help_is_local_and_read_only(assistant_env, command, heading):
    settings, db, client, manager = assistant_env
    await db.update_user_settings("10001", model="gemini-4-argon", context_messages=5,
                                  search_enabled=False, thinking_level="HIGH", max_output_tokens=4096)
    before = await db.get_user_settings("10001")
    manager._pending_images["10001"] = {"images": ["sentinel"], "timestamp": 1}
    reply = await manager.process_message("10001", command)
    assert reply.startswith(heading)
    assert "gemini-4-argon" in reply
    assert "5 条消息（相当于 2.5 轮对话）" in reply
    assert "实时搜索：关闭" in reply
    assert "4096 Token" in reply
    assert client.call_count == 0
    assert await db.get_user_settings("10001") == before
    assert manager._pending_images["10001"]["images"] == ["sentinel"]


@pytest.mark.asyncio
async def test_unknown_help_topic_does_not_become_fuzzy_command(assistant_env):
    settings, db, client, manager = assistant_env
    reply = await manager.handle_command("10001", "//help 不存在的分类")
    assert "未找到帮助分类" in reply
    assert "//help 图片" in reply
    assert client.call_count == 0
    assert await db.get_user_settings("10001") is None


@pytest.mark.asyncio
async def test_full_help_and_aliases_match_without_api(assistant_env):
    settings, db, client, manager = assistant_env
    expected = await manager.handle_command("10001", "//help")
    for command in ("/help", "//帮助", "//?", "//help 全部", "/帮助 all"):
        assert await manager.handle_command("10001", command) == expected
    # The full guide includes the same sections returned by topic help.
    for topic in ("基础", "模型", "思考", "搜索", "记忆", "输出", "图片", "用法"):
        topic_reply = await manager.handle_command("10001", "//help " + topic)
        assert topic_reply.split("\n\n【你当前的配置】")[0] in expected
    assert client.call_count == 0
    assert await db.get_user_settings("10001") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["LOW", "MEDIUM", "HIGH"])
@pytest.mark.parametrize("command,expected", [
    ("//把思考强度设为低", "LOW"),
    ("//思考强度调成中", "MEDIUM"),
    ("//把思考强度设置为高", "HIGH"),
    ("//思考强度 LOW", "LOW"),
    ("//思考强度 medium", "MEDIUM"),
    ("//thinking level high", "HIGH"),
    ("//思考强度从高调到低", "LOW"),
    ("//中思考", "MEDIUM"),
])
async def test_thinking_explicit_target_overrides_current_locally(assistant_env, initial, command, expected):
    settings, db, client, manager = assistant_env
    await db.update_user_settings("10001", thinking_level=initial, model="gemini-3.8-flash")
    reply = await manager.handle_command("10001", command)
    assert f"思考强度={expected}" in reply
    cfg = await db.get_user_settings("10001")
    assert cfg["thinking_level"] == expected
    assert cfg["model"] == "gemini-3.8-flash"
    assert client.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [
    "//思考强度调低一点", "//思考强度更低一点", "//降低思考强度",
    "//思考强度减弱", "//思考再低一点",
])
async def test_thinking_decreases_stepwise_and_clamps(assistant_env, command):
    settings, db, client, manager = assistant_env
    await db.update_user_settings("10001", thinking_level="HIGH")
    for expected in ("MEDIUM", "LOW", "LOW"):
        reply = await manager.handle_command("10001", command)
        assert f"思考强度={expected}" in reply
        assert (await db.get_user_settings("10001"))["thinking_level"] == expected
    assert client.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["//思考强度更高一点", "//提高思考强度"])
async def test_thinking_increases_stepwise_and_bare_noun_has_no_direction(assistant_env, command):
    settings, db, client, manager = assistant_env
    await db.update_user_settings("10001", thinking_level="LOW")
    for expected in ("MEDIUM", "HIGH", "HIGH"):
        reply = await manager.handle_command("10001", command)
        assert f"思考强度={expected}" in reply
    assert manager._parse_command_fast("思考强度", {"thinking_level": "MEDIUM"}) == {}
    assert client.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("command,expected", [
    ("//把思考强度设为低", "LOW"),
    ("//思考强度调成中", "MEDIUM"),
    ("//高思考", "HIGH"),
])
async def test_thinking_setting_reaches_next_model_call(assistant_env, command, expected):
    settings, db, client, manager = assistant_env
    await db.update_user_settings("10001", thinking_level="HIGH")
    await manager.process_message("10001", command)
    await manager.process_message("10001", "请回复测试消息")
    assert client.call_count == 1
    assert client.last_thinking_level == expected


@pytest.mark.asyncio
async def test_command_random_jitter_range(assistant_env, monkeypatch):
    """
    Verify 3-6s random jitter on command processing:
    1. Setting command_jitter_min_seconds=3.0 and command_jitter_max_seconds=6.0
    2. Any command (e.g. //help, //status, //成本, //高思考) causes sleep between 3.0 and 6.0 seconds.
    3. Normal conversation messages do NOT trigger command jitter sleep.
    """
    settings, db, client, manager = assistant_env
    user_id = "10001"

    settings.command_jitter_min_seconds = 3.0
    settings.command_jitter_max_seconds = 6.0

    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    # 1. Test command triggers 3-6s jitter
    reply = await manager.process_message(user_id, "//help")
    assert reply is not None
    assert len(sleep_calls) == 1
    jitter = sleep_calls[0]
    assert 3.0 <= jitter <= 6.0

    sleep_calls.clear()

    # 2. Test slash command triggers 3-6s jitter
    reply2 = await manager.process_message(user_id, "/status")
    assert reply2 is not None
    assert len(sleep_calls) == 1
    assert 3.0 <= sleep_calls[0] <= 6.0

    sleep_calls.clear()

    # 3. Test non-command (regular chat) does NOT trigger command jitter
    sleep_calls.clear()
    reply3 = await manager.process_message(user_id, "你好，请问你是谁？")
    assert reply3 is not None
    # No sleep >= 3.0s should be triggered for regular conversation
    assert not any(s >= 3.0 for s in sleep_calls)
