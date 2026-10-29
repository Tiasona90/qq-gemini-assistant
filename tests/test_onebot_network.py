"""
End-to-End WebSocket Network Test for OneBot 11 Adapter.
Tests real WebSocket client-server handshakes, token authentication,
event dispatching, and reply transmission.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import pytest
import pytest_asyncio
import websockets

from config import Settings
from db import Database
from mock_gemini import MockGeminiClient
from chat_manager import ChatManager
from qq_adapter.onebot_v11 import OneBotV11Adapter


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="qq_ws_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.asyncio
async def test_onebot_reverse_websocket_flow(temp_dir):
    """
    Test OneBot 11 reverse WebSocket flow:
    - Adapter starts server on localhost
    - Client connects and sends a private message event
    - Client receives send_private_msg action
    """
    test_port = 18099
    db_file = os.path.join(temp_dir, "ws_test.db")
    settings = Settings(
        gemini_model="gemini-3.8-flash",
        qq_whitelist_raw="20001",
        reverse_ws_host="127.0.0.1",
        reverse_ws_port=test_port,
        onebot_access_token="secret_token_123",
        sqlite_db_path=db_file,
    )

    db = Database(db_file)
    await db.init_db()
    mock_client = MockGeminiClient()
    chat_mgr = ChatManager(settings, db, mock_client)
    adapter = OneBotV11Adapter(settings, chat_mgr)

    await adapter.start_reverse_server()

    try:
        # Connect simulated NapCat client with token
        headers = {"Authorization": "Bearer secret_token_123"}
        uri = f"ws://127.0.0.1:{test_port}/onebot/v11/ws"

        async with websockets.connect(uri, additional_headers=headers) as ws:
            # Send OneBot 11 private message event
            event = {
                "post_type": "message",
                "message_type": "private",
                "sub_type": "friend",
                "message_id": 998877,
                "user_id": 20001,
                "message": "你好，这是WS端到端测试",
                "raw_message": "你好，这是WS端到端测试",
                "font": 0,
                "sender": {"user_id": 20001, "nickname": "TestUser"},
            }
            await ws.send(json.dumps(event))

            # Wait for response action
            raw_reply = await asyncio.wait_for(ws.recv(), timeout=5.0)
            reply_obj = json.loads(raw_reply)

            assert reply_obj.get("action") == "send_private_msg"
            assert reply_obj.get("params", {}).get("user_id") == 20001
            assert "WS端到端测试" in reply_obj.get("params", {}).get("message")

    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_onebot_unauthorized_token_rejection(temp_dir):
    """Test that unauthorized WebSocket client is rejected with 4001."""
    test_port = 18098
    db_file = os.path.join(temp_dir, "ws_test2.db")
    settings = Settings(
        reverse_ws_host="127.0.0.1",
        reverse_ws_port=test_port,
        onebot_access_token="correct_token",
        sqlite_db_path=db_file,
    )

    db = Database(db_file)
    await db.init_db()
    mock_client = MockGeminiClient()
    chat_mgr = ChatManager(settings, db, mock_client)
    adapter = OneBotV11Adapter(settings, chat_mgr)

    await adapter.start_reverse_server()

    try:
        # Connect with invalid token
        headers = {"Authorization": "Bearer WRONG_TOKEN"}
        uri = f"ws://127.0.0.1:{test_port}/onebot/v11/ws"

        with pytest.raises(Exception):
            async with websockets.connect(uri, additional_headers=headers) as ws:
                await ws.recv()

    finally:
        await adapter.stop()
