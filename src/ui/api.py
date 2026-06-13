"""
Lightweight HTTP API for embedding OnIt chat in external web UIs.

Provides JSON + SSE endpoints that mirror the CLI / web chat feature set:
session management, streaming responses, stop/interrupt, file upload, and
history replay from per-session JSONL files.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the OnIt chat API. "
        "Install it with: pip install fastapi uvicorn"
    ) from exc


_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)


@dataclass
class ApiSession:
    """Per-client chat session state."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_path: str = ""
    data_path: str = ""
    processing: bool = False
    streaming_content: str = ""
    streaming_active: bool = False
    safety_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=10))
    created: datetime = field(default_factory=datetime.now)


class ChatAPIServer:
    """FastAPI server exposing OnIt chat for external UIs."""

    def __init__(
        self,
        onit: Any,
        port: int = 9002,
        host: str = "0.0.0.0",
        session_path: str | None = None,
        show_logs: bool = False,
        cors_origins: list[str] | None = None,
    ) -> None:
        self._onit = onit
        self.port = port
        self.host = host
        self.show_logs = show_logs
        self.session_path = session_path or onit.config_data.get(
            'session_path', '~/.onit/sessions')
        self._sessions: dict[str, ApiSession] = {}
        self._app = self._build_app(cors_origins or ["*"])

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _sessions_dir(self) -> str:
        base = self.session_path
        if base.endswith('.jsonl'):
            return os.path.dirname(os.path.expanduser(base))
        return os.path.expanduser(base)

    def _get_or_create_session(
        self, session_id: str | None = None,
    ) -> tuple[str, ApiSession]:
        if session_id and not _UUID_RE.match(session_id):
            session_id = None

        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]

        sessions_dir = self._sessions_dir()
        os.makedirs(sessions_dir, exist_ok=True)

        if session_id:
            session_file = os.path.join(sessions_dir, f"{session_id}.jsonl")
            if os.path.exists(session_file):
                session = ApiSession(session_id=session_id)
                session.session_path = session_file
                session.data_path = str(
                    Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
                os.makedirs(session.data_path, exist_ok=True)
                self._sessions[session_id] = session
                return session_id, session

        session = ApiSession(session_id=session_id) if session_id else ApiSession()
        session.session_path = os.path.join(
            sessions_dir, f"{session.session_id}.jsonl")
        if not os.path.exists(session.session_path):
            with open(session.session_path, "w", encoding="utf-8") as handle:
                handle.write("")
        session.data_path = str(
            Path(tempfile.gettempdir()) / "onit" / "data" / session.session_id)
        os.makedirs(session.data_path, exist_ok=True)
        self._sessions[session.session_id] = session

        now = datetime.now()
        expired = [
            sid for sid, item in self._sessions.items()
            if (now - item.created) > timedelta(hours=24)
        ]
        for sid in expired:
            del self._sessions[sid]

        return session.session_id, session

    def _load_history(self, session_path: str, max_turns: int = 100) -> list[dict]:
        history: list[dict] = []
        try:
            if os.path.exists(session_path):
                with open(session_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        task = entry.get("task", "").strip()
                        response = entry.get("response", "").strip()
                        if task:
                            history.append({"role": "user", "content": task})
                        if response:
                            history.append({"role": "assistant", "content": response})
        except OSError:
            pass
        return history[-max_turns * 2:]

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def _run_task(
        self,
        session: ApiSession,
        message: str,
        on_event: Any | None = None,
    ) -> str:
        session.processing = True
        session.streaming_content = ""
        session.streaming_active = False

        try:
            def _on_token(_token: str, full_content: str) -> None:
                session.streaming_content = full_content
                session.streaming_active = True
                if on_event:
                    on_event("token", {"content": full_content})

            def _on_complete(_content: str, tok_s: float) -> None:
                session.streaming_active = False
                if on_event:
                    on_event("stream_end", {"tokens_per_second": tok_s})

            stats: dict[str, Any] = {}
            started = time.monotonic()
            response = await self._onit.process_task(
                message,
                session_path=session.session_path,
                data_path=session.data_path,
                safety_queue=session.safety_queue,
                stream_callback=_on_token,
                stream_complete_callback=_on_complete,
                stats=stats,
                session_id=session.session_id,
            )
            elapsed = time.monotonic() - started
            if on_event:
                on_event("complete", {
                    "response": response or "",
                    "elapsed_s": round(elapsed, 2),
                    "tokens_per_second": stats.get("tokens_per_second", 0),
                })
            return response or ""
        except Exception as exc:
            logger.error("Chat API task failed: %s", exc)
            if on_event:
                on_event("error", {"message": str(exc)})
            raise
        finally:
            session.processing = False
            session.streaming_active = False
            session.streaming_content = ""

    # ------------------------------------------------------------------
    # FastAPI routes
    # ------------------------------------------------------------------

    def _build_app(self, cors_origins: list[str]) -> FastAPI:
        app = FastAPI(title="OnIt Chat API", version="1.0.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class _DisconnectMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, outer: "ChatAPIServer"):
                super().__init__(app)
                self._outer = outer

            async def dispatch(self, request: Request, call_next):
                try:
                    return await call_next(request)
                except asyncio.CancelledError:
                    sid = request.headers.get("X-OnIt-Session")
                    if sid and sid in self._outer._sessions:
                        session = self._outer._sessions[sid]
                        if not session.safety_queue.full():
                            session.safety_queue.put_nowait(True)
                    raise

        app.add_middleware(_DisconnectMiddleware, outer=self)

        @app.get("/api/v1/health")
        async def health() -> JSONResponse:
            return JSONResponse({
                "ok": True,
                "status": getattr(self._onit, "status", "unknown"),
                "tools": len(getattr(self._onit, "tool_registry", {}) or {}),
            })

        @app.post("/api/v1/chat/session")
        async def create_session() -> JSONResponse:
            session_id, session = self._get_or_create_session()
            return JSONResponse({
                "ok": True,
                "session_id": session_id,
                "processing": session.processing,
            })

        @app.get("/api/v1/chat/history")
        async def history(session_id: str) -> JSONResponse:
            _, session = self._get_or_create_session(session_id)
            return JSONResponse({
                "ok": True,
                "session_id": session.session_id,
                "messages": self._load_history(session.session_path),
                "processing": session.processing,
                "streaming_content": session.streaming_content
                if session.streaming_active else "",
            })

        @app.get("/api/v1/chat/status")
        async def status(session_id: str) -> JSONResponse:
            _, session = self._get_or_create_session(session_id)
            return JSONResponse({
                "ok": True,
                "session_id": session.session_id,
                "processing": session.processing,
                "streaming_active": session.streaming_active,
                "streaming_content": session.streaming_content,
            })

        @app.post("/api/v1/chat/stop")
        async def stop(request: Request) -> JSONResponse:
            body = await request.json()
            session_id = body.get("session_id")
            if not session_id:
                return JSONResponse({"ok": False, "error": "session_id required"}, 400)
            _, session = self._get_or_create_session(session_id)
            if not session.safety_queue.full():
                session.safety_queue.put_nowait(True)
            return JSONResponse({"ok": True})

        @app.post("/api/v1/chat/message")
        async def message(request: Request):
            body = await request.json()
            session_id = body.get("session_id")
            text = (body.get("message") or "").strip()
            stream = body.get("stream", True)
            if not text:
                return JSONResponse({"ok": False, "error": "message required"}, 400)

            _, session = self._get_or_create_session(session_id)
            if session.processing:
                return JSONResponse(
                    {"ok": False, "error": "session is busy"}, status_code=409)

            if not stream:
                try:
                    response = await self._run_task(session, text)
                except Exception as exc:
                    return JSONResponse({"ok": False, "error": str(exc)}, 500)
                return JSONResponse({
                    "ok": True,
                    "session_id": session.session_id,
                    "response": response,
                })

            queue: asyncio.Queue = asyncio.Queue()

            def _enqueue(event_type: str, payload: dict) -> None:
                queue.put_nowait((event_type, payload))

            async def _worker() -> None:
                try:
                    await self._run_task(session, text, on_event=_enqueue)
                except Exception as exc:
                    await queue.put(("error", {"message": str(exc)}))
                finally:
                    await queue.put(("_done", {}))

            asyncio.create_task(_worker())

            async def _sse():
                yield "event: started\ndata: {}\n\n"
                while True:
                    event_type, payload = await queue.get()
                    if event_type == "_done":
                        break
                    data = json.dumps(payload, ensure_ascii=False)
                    yield f"event: {event_type}\ndata: {data}\n\n"

            return StreamingResponse(
                _sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-OnIt-Session": session.session_id,
                },
            )

        @app.post("/api/v1/chat/upload")
        async def upload(
            session_id: str = Form(...),
            file: UploadFile = File(...),
        ) -> JSONResponse:
            _, session = self._get_or_create_session(session_id)
            os.makedirs(session.data_path, exist_ok=True)
            safe_name = os.path.basename(file.filename or "upload.bin")
            dest = os.path.join(session.data_path, safe_name)
            content = await file.read()
            with open(dest, "wb") as handle:
                handle.write(content)
            return JSONResponse({
                "ok": True,
                "filename": safe_name,
                "path": dest,
            })

        @app.get("/uploads/{session_id}/{filename}")
        async def serve_upload(session_id: str, filename: str):
            _, session = self._get_or_create_session(session_id)
            safe_name = os.path.basename(filename)
            filepath = os.path.join(session.data_path, safe_name)
            if not os.path.isfile(filepath):
                return JSONResponse({"ok": False, "error": "not found"}, 404)
            return FileResponse(filepath, filename=safe_name)

        return app

    async def serve(self) -> None:
        import uvicorn

        _verbose = self._onit.verbose or self.show_logs
        log_level = "info" if _verbose else "warning"
        print(
            f"OnIt Chat API running at http://{self.host}:{self.port}/ "
            f"(Ctrl+C to stop)")
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level=log_level,
            access_log=_verbose,
        )
        server = uvicorn.Server(config)
        await server.serve()
