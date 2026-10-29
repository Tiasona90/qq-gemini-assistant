"""Regression coverage for OneBot's structured/CQ image representations."""
import io
import json
from unittest.mock import AsyncMock

import httpx
from PIL import Image
import pytest

from image_utils import extract_message_and_images
from config import Settings
from db import Database
from chat_manager import ChatManager
from mock_gemini import MockGeminiClient
from qq_adapter.onebot_v11 import OneBotV11Adapter


URL = "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=test-file&rkey=test-key"
CQ_URL = URL.replace("&", "&amp;")


def test_structured_image_not_duplicated_by_raw_cq():
    event = {
        "message": [{"type": "image", "data": {"url": URL, "file": "test.jpg"}}],
        "raw_message": f"[CQ:image,file=test.jpg,url={CQ_URL}]",
    }
    assert extract_message_and_images(event) == ("", [URL])


@pytest.mark.parametrize("field", ["message", "raw_message"])
def test_cq_string_decodes_url_and_text(field):
    event = {field: f"看看 &amp; &#91;原图&#93;[CQ:image,file=test.jpg,url={CQ_URL}]"}
    assert extract_message_and_images(event) == ("看看 & [原图]", [URL])


def test_cq_parameter_commas_brackets_and_signed_url_preserved():
    source = "https://example.test/image?name=a,b[c]&signature=a%2Bb%3D&notit=ok"
    encoded = source.replace("&", "&amp;").replace(",", "&#44;").replace("[", "&#91;").replace("]", "&#93;")
    assert extract_message_and_images({"message": f"[CQ:image,url={encoded}]"}) == ("", [source])


def test_cq_decodes_only_one_layer():
    event = {"message": "[CQ:image,url=https://example.test/a?literal=&amp;#44;]"}
    assert extract_message_and_images(event)[1] == ["https://example.test/a?literal=&#44;"]


def test_two_real_images_retained_once_in_order():
    second = URL.replace("test-file", "second-file")
    event = {
        "message": [{"type": "image", "data": {"url": ref}} for ref in (URL, second, URL)],
        "raw_message": f"[CQ:image,url={CQ_URL}][CQ:image,url={second.replace('&', '&amp;')}]",
    }
    assert extract_message_and_images(event) == ("", [URL, second])


def test_raw_images_used_when_segments_contain_only_text():
    event = {"message": [{"type": "text", "data": {"text": "分析图片"}}],
             "raw_message": f"分析图片[CQ:image,url={CQ_URL}]"}
    assert extract_message_and_images(event) == ("分析图片", [URL])


def test_malformed_segments_and_file_fallback():
    event = {"message": [None, {}, {"type": "image", "data": None},
                         {"type": "image", "data": {"file": "base64://YWJj"}}]}
    assert extract_message_and_images(event) == ("", ["base64://YWJj"])


@pytest.mark.asyncio
@pytest.mark.parametrize("with_text", [False, True])
async def test_onebot_image_download_and_gemini_delivery(tmp_path, monkeypatch, with_text):
    """Replay duplicate structured/CQ references without contacting Tencent or Gemini."""
    image = Image.new("RGB", (64, 48), (10, 80, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    requests = []

    def respond(request):
        requests.append(str(request.url))
        assert str(request.url) == URL
        return httpx.Response(200, content=buffer.getvalue())

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs))
    settings = Settings(gemini_api_key="MOCK", qq_whitelist_raw="10001", enable_search_default=False,
                        sqlite_db_path=str(tmp_path / "images.db"))
    db = Database(settings.sqlite_db_path)
    await db.init_db()
    client = MockGeminiClient(simulate_delay=0)
    manager = ChatManager(settings, db, client)
    adapter = OneBotV11Adapter(settings, manager)
    ws = AsyncMock()
    segments = [{"type": "image", "data": {"url": URL}}]
    if with_text:
        segments.append({"type": "text", "data": {"text": "图片里有什么？"}})
    event = {"post_type": "message", "message_type": "private", "user_id": 10001,
             "message_id": 100, "message": segments,
             "raw_message": f"[CQ:image,url={CQ_URL}]" + ("图片里有什么？" if with_text else "")}
    await adapter._handle_message_event(event, ws)
    assert requests == [URL]
    first_reply = json.loads(ws.send.call_args.args[0])["params"]["message"]
    if not with_text:
        assert "已收到你的图片" in first_reply
        assert len(manager._pending_images["10001"]["images"]) == 1
        assert client.call_count == 0
        await adapter._handle_message_event({"post_type": "message", "message_type": "private",
            "user_id": 10001, "message_id": 101, "message": "图片里有什么？"}, ws)
    assert len(client.last_images) == 1
    assert client.last_images[0]["final_size"] == (64, 48)
    assert manager._pending_images == {}
    assert "已识别你发送的 1 张图片" in json.loads(ws.send.call_args.args[0])["params"]["message"]

