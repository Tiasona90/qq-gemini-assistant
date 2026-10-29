"""
OneBot 11 WebSocket Adapter for NapCat / Lagrange / Go-CQHttp.
Supports both Reverse WebSocket Server (NapCat connects to bot)
and Forward WebSocket Client (bot connects to NapCat).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Set, Any
import websockets


from config import Settings
from chat_manager import ChatManager
from image_utils import extract_message_and_images

logger = logging.getLogger(__name__)


class OneBotV11Adapter:
    def __init__(self, settings: Settings, chat_manager: ChatManager) -> None:
        self.settings = settings
        self.chat_manager = chat_manager
        self.active_connections: Set[Any] = set()
        self._server = None
        self._is_running = False

    async def _handle_message_event(
        self,
        event: dict,
        ws: Any,
    ) -> None:
        """
        Process incoming OneBot v11 event payload.
        """
        post_type = event.get("post_type")

        # We only handle message events
        if post_type != "message":
            return

        message_type = event.get("message_type")
        user_id = str(event.get("user_id", "")).strip()
        message_id = str(event.get("message_id", ""))

        # 1. Strictly ignore group messages (as specified: reject stranger and group messages)
        if message_type != "private":
            logger.debug(f"[OneBot] Ignored non-private message type: {message_type}")
            return

        # 2. Extract plain text content and image sources
        clean_text, image_sources = extract_message_and_images(event)

        if not clean_text and not image_sources:
            return

        logger.info(
            f"[OneBot] Received private message from user {user_id}: "
            f"text={clean_text[:50]!r}, images_count={len(image_sources)}"
        )

        # 3. Delegate to ChatManager
        reply = await self.chat_manager.process_message(
            user_id=user_id,
            content=clean_text,
            message_id=message_id,
            image_sources=image_sources,
        )

        if reply:
            await self._send_private_message(ws, user_id, reply)

    async def _send_private_message(
        self,
        ws: Any,
        user_id: str,
        content: str,
    ) -> None:
        """
        Send OneBot 11 'send_private_msg' action back through WebSocket.
        """
        payload = {
            "action": "send_private_msg",
            "params": {
                "user_id": int(user_id) if user_id.isdigit() else user_id,
                "message": content,
            },
            "echo": f"reply_{user_id}_{int(asyncio.get_event_loop().time())}",
        }
        try:
            target_ws = ws
            if getattr(target_ws, "closed", False) and self.active_connections:
                target_ws = next(iter(self.active_connections))
            await target_ws.send(json.dumps(payload, ensure_ascii=False))
            logger.info(f"[OneBot] Sent reply to user {user_id}")
        except Exception as e:
            logger.error(f"[OneBot] Failed to send reply to user {user_id}: {e}")

    # ============================================================
    # Reverse WebSocket Mode (Server)
    # ============================================================
    def _process_request(self, connection: Any, request: Any) -> Optional[Any]:
        """
        Validate incoming WebSocket handshake requests against access token.
        """
        if not self.settings.onebot_access_token:
            return None

        expected_token = self.settings.onebot_access_token.strip()
        auth = request.headers.get("Authorization", "").strip()
        header_token = request.headers.get("access_token", "").strip()
        path = getattr(request, "path", "")
        query_token = ""
        if "access_token=" in path:
            query_token = path.split("access_token=")[-1].split("&")[0].strip()

        valid = (
            auth == f"Bearer {expected_token}"
            or auth == expected_token
            or header_token == expected_token
            or query_token == expected_token
        )

        if not valid:
            logger.warning("[OneBot] Unauthorized WebSocket connection attempt rejected.")
            import http
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")

        return None

    async def _reverse_ws_handler(self, ws: Any, *args: Any) -> None:
        """
        Handles incoming connection from NapCat (acting as WebSocket client).
        """
        self.active_connections.add(ws)
        remote = getattr(ws, "remote_address", "unknown")
        logger.info(f"[OneBot] NapCat Reverse WebSocket client connected from {remote}")

        try:
            async for raw_data in ws:
                try:
                    event = json.loads(raw_data)
                    asyncio.create_task(self._handle_message_event(event, ws))
                except json.JSONDecodeError:
                    logger.warning("[OneBot] Received invalid JSON payload")
        except websockets.ConnectionClosed:
            logger.info(f"[OneBot] Reverse WebSocket client {remote} disconnected")
        finally:
            self.active_connections.discard(ws)

    async def start_reverse_server(self) -> None:
        """Start listening for NapCat connections."""
        host = self.settings.reverse_ws_host
        port = self.settings.reverse_ws_port
        logger.info(f"[OneBot] Starting Reverse WebSocket Server on ws://{host}:{port}{self.settings.reverse_ws_path}")
        self._server = await websockets.serve(
            self._reverse_ws_handler,
            host=host,
            port=port,
            process_request=self._process_request,
        )

    # ============================================================
    # Forward WebSocket Mode (Client)
    # ============================================================
    async def start_forward_client(self) -> None:
        """Connect out to NapCat's WebSocket Server with auto-reconnect."""
        url = self.settings.forward_ws_url
        headers = {}
        if self.settings.onebot_access_token:
            headers["Authorization"] = f"Bearer {self.settings.onebot_access_token}"

        logger.info(f"[OneBot] Starting Forward WebSocket Client connecting to {url}")
        while self._is_running:
            try:
                connect_kwargs = {}
                if headers:
                    connect_kwargs["additional_headers"] = headers
                async with websockets.connect(url, **connect_kwargs) as ws:
                    self.active_connections.add(ws)
                    logger.info(f"[OneBot] Connected to OneBot server at {url}")
                    async for raw_data in ws:
                        try:
                            event = json.loads(raw_data)
                            asyncio.create_task(self._handle_message_event(event, ws))
                        except json.JSONDecodeError:
                            logger.warning("[OneBot] Received invalid JSON payload")
            except (websockets.ConnectionClosed, OSError) as e:
                logger.warning(f"[OneBot] Forward connection disconnected or failed ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"[OneBot] Unexpected error in forward WS client: {e}")
                await asyncio.sleep(5)

    async def start(self) -> None:
        """Start adapter based on configured mode."""
        self._is_running = True
        if self.settings.onebot_mode == "forward_ws":
            asyncio.create_task(self.start_forward_client())
        else:
            await self.start_reverse_server()

    async def stop(self) -> None:
        """Gracefully stop adapter and close all connections."""
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for ws in list(self.active_connections):
            await ws.close()
        self.active_connections.clear()
        logger.info("[OneBot] Adapter stopped.")
