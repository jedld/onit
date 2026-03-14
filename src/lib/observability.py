"""Observability helpers for tracing OnIt runtime behavior.

Provides:
- structured JSONL event recording for LLM and MCP interactions
- aggregate runtime stats for external introspection
- lightweight probe helpers for model and MCP endpoints
- an optional FastAPI server exposing traces and probe data
"""

from __future__ import annotations

import copy
import json
import logging
import os
import socket
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:  # pragma: no cover - fastapi is a project dependency
    FastAPI = None
    Query = None
    HTMLResponse = None
    uvicorn = None

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any, limit: int = 2000) -> Any:
    """Return a JSON-serializable, size-bounded representation."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, (list, tuple)):
        return [_safe_json(item, limit=limit) for item in value[:50]]
    if isinstance(value, dict):
        sanitized = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 50:
                sanitized["__truncated__"] = True
                break
            sanitized[str(key)] = _safe_json(item, limit=limit)
        return sanitized
    try:
        text = json.dumps(value)
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _content_metrics(messages: list[dict[str, Any]]) -> dict[str, int]:
    text_chars = 0
    image_parts = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_chars += len(item.get("text", ""))
                elif item.get("type") == "image_url":
                    image_parts += 1
    return {"text_chars": text_chars, "image_parts": image_parts}


def _probe_endpoint(url: str, timeout: float = 1.0) -> dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    started = time.monotonic()
    try:
        connection = socket.create_connection((host, port), timeout=timeout)
        connection.close()
        latency_ms = round((time.monotonic() - started) * 1000, 2)
        return {
            "url": url,
            "host": host,
            "port": port,
            "reachable": True,
            "latency_ms": latency_ms,
        }
    except OSError as exc:
        latency_ms = round((time.monotonic() - started) * 1000, 2)
        return {
            "url": url,
            "host": host,
            "port": port,
            "reachable": False,
            "latency_ms": latency_ms,
            "error": str(exc),
        }


@dataclass
class ProbeReport:
    model: dict[str, Any]
    mcp_servers: list[dict[str, Any]]
    tool_registry: dict[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "model": self.model,
            "mcp_servers": self.mcp_servers,
            "tool_registry": self.tool_registry,
        }


class TraceRecorder:
    """Collects structured events and summary stats for OnIt runtime activity."""

    def __init__(self,
                 base_dir: str,
                 enabled: bool = True,
                 max_events: int = 2000,
                 file_name: str = "events.jsonl") -> None:
        self.enabled = enabled
        self.base_dir = os.path.expanduser(base_dir)
        self.max_events = max(100, int(max_events))
        self.file_name = file_name
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._sequence = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._stats = {
            "llm_requests": 0,
            "llm_errors": 0,
            "mcp_requests": 0,
            "mcp_errors": 0,
            "tool_decisions": 0,
            "sessions": Counter(),
            "mcp_tools": Counter(),
            "llm_models": Counter(),
            "latency_ms": {
                "llm": deque(maxlen=500),
                "mcp": deque(maxlen=500),
            },
            "last_error": None,
            "active_tasks": {},
            "completed_tasks": deque(maxlen=200),
            "active_operations": {},
            "completed_operations": deque(maxlen=400),
        }
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        self.events_path = os.path.join(self.base_dir, self.file_name)

    @staticmethod
    def _extract_task_summary(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": event.get("task_id"),
            "session_id": event.get("session_id"),
            "status": event.get("status", "inflight"),
            "task_preview": event.get("task_preview"),
            "timestamp": event.get("timestamp"),
            "duration_ms": event.get("duration_ms"),
            "response_preview": event.get("response_preview"),
            "reason": event.get("reason") or event.get("error"),
            "image_count": event.get("image_count"),
        }

    @staticmethod
    def _extract_operation_summary(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "operation_id": event.get("operation_id"),
            "task_id": event.get("task_id"),
            "session_id": event.get("session_id"),
            "operation_kind": event.get("operation_kind"),
            "status": event.get("status", "inflight"),
            "name": event.get("tool_name") or event.get("model"),
            "url": event.get("url") or event.get("host"),
            "timestamp": event.get("timestamp"),
            "latency_ms": event.get("latency_ms"),
            "content_preview": event.get("content_preview") or event.get("result_preview"),
            "error": event.get("error"),
            "finish_reason": event.get("finish_reason"),
            "kind": event.get("kind"),
        }

    def _update_task_state(self, event: dict[str, Any]) -> None:
        task_id = event.get("task_id")
        if not task_id:
            return

        active_tasks = self._stats["active_tasks"]
        completed_tasks = self._stats["completed_tasks"]
        event_type = event["event_type"]

        if event_type == "agent.task_received":
            summary = self._extract_task_summary(event)
            summary["status"] = "inflight"
            summary["started_at"] = event.get("timestamp")
            active_tasks[task_id] = summary
            return

        existing = active_tasks.get(task_id, {})
        if event_type in {"agent.task_completed", "agent.task_failed", "agent.task_cancelled"}:
            summary = self._extract_task_summary(event)
            summary["started_at"] = existing.get("started_at") or existing.get("timestamp")
            summary["task_preview"] = summary.get("task_preview") or existing.get("task_preview")
            summary["session_id"] = summary.get("session_id") or existing.get("session_id")
            if event_type == "agent.task_completed":
                summary["status"] = "completed"
            elif event_type == "agent.task_failed":
                summary["status"] = "failed"
            else:
                summary["status"] = "cancelled"
            active_tasks.pop(task_id, None)
            completed_tasks.appendleft(summary)

    def _update_operation_state(self, event: dict[str, Any]) -> None:
        operation_id = event.get("operation_id")
        if not operation_id:
            return

        active_operations = self._stats["active_operations"]
        completed_operations = self._stats["completed_operations"]
        event_type = event["event_type"]

        if event_type in {"llm.request", "mcp.request"}:
            summary = self._extract_operation_summary(event)
            summary["status"] = "inflight"
            summary["started_at"] = event.get("timestamp")
            active_operations[operation_id] = summary
            return

        existing = active_operations.get(operation_id, {})
        if event_type in {"llm.response", "llm.error", "mcp.response", "mcp.error"}:
            summary = self._extract_operation_summary(event)
            summary["started_at"] = existing.get("started_at") or existing.get("timestamp")
            summary["name"] = summary.get("name") or existing.get("name")
            summary["url"] = summary.get("url") or existing.get("url")
            summary["task_id"] = summary.get("task_id") or existing.get("task_id")
            summary["session_id"] = summary.get("session_id") or existing.get("session_id")
            if event_type.endswith("response"):
                summary["status"] = "completed"
            else:
                summary["status"] = "failed"
            active_operations.pop(operation_id, None)
            completed_operations.appendleft(summary)

    def record(self, event_type: str, **payload: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        operation_kind = payload.get("operation_kind")
        if operation_kind is None:
            if event_type.startswith("llm."):
                operation_kind = "llm"
            elif event_type.startswith("mcp."):
                operation_kind = "mcp"

        event = {
            "timestamp": _utc_now(),
            "event_type": event_type,
            "operation_kind": operation_kind,
            **{key: _safe_json(value) for key, value in payload.items()},
        }

        with self._lock:
            self._sequence += 1
            event["sequence"] = self._sequence
            self._events.append(event)
            session_id = event.get("session_id")
            if session_id:
                self._stats["sessions"][session_id] += 1
            if event_type == "llm.request":
                self._stats["llm_requests"] += 1
                model = event.get("model")
                if model:
                    self._stats["llm_models"][model] += 1
            elif event_type == "llm.error":
                self._stats["llm_errors"] += 1
                self._stats["last_error"] = event
            elif event_type == "agent.tool_decision":
                self._stats["tool_decisions"] += 1
            elif event_type == "mcp.request":
                self._stats["mcp_requests"] += 1
                tool_name = event.get("tool_name")
                if tool_name:
                    self._stats["mcp_tools"][tool_name] += 1
            elif event_type == "mcp.error":
                self._stats["mcp_errors"] += 1
                self._stats["last_error"] = event

            latency_ms = event.get("latency_ms")
            if isinstance(latency_ms, (int, float)):
                if event_type.startswith("llm."):
                    self._stats["latency_ms"]["llm"].append(latency_ms)
                elif event_type.startswith("mcp."):
                    self._stats["latency_ms"]["mcp"].append(latency_ms)

            self._update_task_state(event)
            self._update_operation_state(event)

            try:
                with open(self.events_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=True) + "\n")
            except OSError as exc:
                logger.warning("Failed to persist trace event: %s", exc)

        return event

    def recent_events(self,
                      limit: int = 50,
                      session_id: str | None = None,
                      event_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        if session_id:
            events = [event for event in events if event.get("session_id") == session_id]
        if event_type:
            events = [event for event in events if event.get("event_type") == event_type]
        return events[-max(1, min(limit, 500)):]

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            llm_latencies = list(self._stats["latency_ms"]["llm"])
            mcp_latencies = list(self._stats["latency_ms"]["mcp"])
            last_error = copy.deepcopy(self._stats["last_error"])
            session_counts = dict(self._stats["sessions"].most_common(20))
            tool_counts = dict(self._stats["mcp_tools"].most_common(20))
            model_counts = dict(self._stats["llm_models"].most_common(20))
            inflight_tasks = len(self._stats["active_tasks"])
            inflight_operations = len(self._stats["active_operations"])
            snapshot = {
                "enabled": self.enabled,
                "base_dir": self.base_dir,
                "events_path": self.events_path,
                "uptime_seconds": round(time.time() - self._started_at, 2),
                "event_count": len(self._events),
                "llm_requests": self._stats["llm_requests"],
                "llm_errors": self._stats["llm_errors"],
                "mcp_requests": self._stats["mcp_requests"],
                "mcp_errors": self._stats["mcp_errors"],
                "tool_decisions": self._stats["tool_decisions"],
                "inflight_task_count": inflight_tasks,
                "completed_task_count": len(self._stats["completed_tasks"]),
                "inflight_operation_count": inflight_operations,
                "completed_operation_count": len(self._stats["completed_operations"]),
                "session_activity": session_counts,
                "tool_activity": tool_counts,
                "model_activity": model_counts,
                "llm_avg_latency_ms": round(sum(llm_latencies) / len(llm_latencies), 2) if llm_latencies else None,
                "mcp_avg_latency_ms": round(sum(mcp_latencies) / len(mcp_latencies), 2) if mcp_latencies else None,
                "last_error": last_error,
            }
        return snapshot

    def dashboard_snapshot(self) -> dict[str, Any]:
        with self._lock:
            inflight_tasks = list(self._stats["active_tasks"].values())
            completed_tasks = list(self._stats["completed_tasks"])
            inflight_operations = list(self._stats["active_operations"].values())
            completed_operations = list(self._stats["completed_operations"])

        ros_operations = [
            op for op in inflight_operations + completed_operations[:50]
            if (op.get("url") and "ros2" in str(op.get("url")).lower())
            or (op.get("name") and "ros" in str(op.get("name")).lower())
        ]

        return {
            "generated_at": _utc_now(),
            "stats": self.stats_snapshot(),
            "inflight_tasks": inflight_tasks[:50],
            "completed_tasks": completed_tasks[:50],
            "inflight_operations": inflight_operations[:100],
            "completed_operations": completed_operations[:100],
            "recent_events": self.recent_events(limit=100),
            "ros_activity": ros_operations[:100],
        }

    def build_probe_report(self,
                           model_host: str | None,
                           mcp_servers: list[dict[str, Any]] | None,
                           tool_registry: Any | None = None) -> ProbeReport:
        model_probe = _probe_endpoint(model_host) if model_host else {
            "url": None,
            "reachable": False,
            "error": "No model host configured",
        }
        server_probes = []
        for server in mcp_servers or []:
            if not server.get("enabled", True):
                continue
            probe = _probe_endpoint(server.get("url", ""))
            probe["name"] = server.get("name")
            server_probes.append(probe)

        registry_summary = {
            "tool_count": len(tool_registry) if tool_registry is not None else 0,
            "tools": sorted(list(tool_registry.tools)) if getattr(tool_registry, "tools", None) else [],
        }
        return ProbeReport(
            model=model_probe,
            mcp_servers=server_probes,
            tool_registry=registry_summary,
            generated_at=_utc_now(),
        )

    def summarize_llm_request(self,
                              session_id: str,
                              task_id: str | None,
                              operation_id: str,
                              model: str,
                              host: str,
                              iteration: int,
                              messages: list[dict[str, Any]],
                              tool_count: int) -> dict[str, Any]:
        metrics = _content_metrics(messages)
        return self.record(
            "llm.request",
            session_id=session_id,
            task_id=task_id,
            operation_id=operation_id,
            model=model,
            host=host,
            iteration=iteration,
            message_count=len(messages),
            tool_count=tool_count,
            text_chars=metrics["text_chars"],
            image_parts=metrics["image_parts"],
        ) or {}

    def summarize_llm_response(self,
                               session_id: str,
                               task_id: str | None,
                               operation_id: str,
                               model: str,
                               iteration: int,
                               latency_ms: float,
                               finish_reason: str | None,
                               content_preview: str | None,
                               tool_call_count: int) -> dict[str, Any]:
        return self.record(
            "llm.response",
            session_id=session_id,
            task_id=task_id,
            operation_id=operation_id,
            model=model,
            iteration=iteration,
            latency_ms=round(latency_ms, 2),
            finish_reason=finish_reason,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
        ) or {}

    def summarize_llm_error(self,
                            session_id: str,
                            task_id: str | None,
                            operation_id: str | None,
                            model: str,
                            iteration: int,
                            latency_ms: float | None,
                            error: str) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "operation_id": operation_id,
            "model": model,
            "iteration": iteration,
            "error": error,
        }
        if latency_ms is not None:
            payload["latency_ms"] = round(latency_ms, 2)
        return self.record("llm.error", **payload) or {}


class IntrospectionServer:
    """Background HTTP server exposing traces, stats, and probe data."""

    def __init__(self,
                 trace_recorder: TraceRecorder,
                 onit_ref: Any,
                 host: str = "127.0.0.1",
                 port: int = 9100) -> None:
        self.trace_recorder = trace_recorder
        self.onit_ref = onit_ref
        self.host = host
        self.port = int(port)
        self._thread: threading.Thread | None = None
        self._server = None

    def start(self) -> None:
        if self._thread is not None or FastAPI is None or uvicorn is None:
            return

        app = FastAPI(title="OnIt Introspection", version="1.0.0")

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {
                "status": "ok",
                "trace": self.trace_recorder.stats_snapshot(),
            }

        @app.get("/probe")
        async def probe() -> dict[str, Any]:
            return self.onit_ref.get_probe_report()

        @app.get("/trace/stats")
        async def trace_stats() -> dict[str, Any]:
            return self.trace_recorder.stats_snapshot()

        @app.get("/trace/events")
        async def trace_events(limit: int = Query(default=50, ge=1, le=500),
                               session_id: str | None = None,
                               event_type: str | None = None) -> dict[str, Any]:
            return {
                "events": self.trace_recorder.recent_events(
                    limit=limit,
                    session_id=session_id,
                    event_type=event_type,
                )
            }

        @app.get("/dashboard/data")
        async def dashboard_data() -> dict[str, Any]:
            return self.trace_recorder.dashboard_snapshot()

        @app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard() -> str:
            return _dashboard_html()

        def _run() -> None:
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            self._server = uvicorn.Server(config)
            self._server.run()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()


def _dashboard_html() -> str:
        return """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>OnIt ROS/MCP Dashboard</title>
    <style>
        :root {
            --bg: #0f1720;
            --panel: #17212b;
            --panel-2: #223142;
            --text: #e6edf4;
            --muted: #9fb1c1;
            --accent: #6ee7b7;
            --warn: #fbbf24;
            --bad: #fb7185;
            --good: #34d399;
            --line: #2f4153;
            --chip: #101923;
        }
        body {
            margin: 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace;
            background: radial-gradient(circle at top, #132131 0%, var(--bg) 55%);
            color: var(--text);
        }
        .wrap {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }
        h1, h2 { margin: 0 0 12px; }
        p { color: var(--muted); margin: 0; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 20px 0 24px;
        }
        .card, .table-card {
            background: rgba(23, 33, 43, 0.92);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18);
        }
        .card { padding: 16px; }
        .metric { font-size: 28px; font-weight: 700; margin-top: 10px; }
        .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
        .row {
            display: grid;
            grid-template-columns: 1.1fr 1.4fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        .row-full { margin-bottom: 16px; }
        .table-card { overflow: hidden; }
        .table-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 16px;
            border-bottom: 1px solid var(--line);
            background: rgba(34, 49, 66, 0.7);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(47, 65, 83, 0.55);
            vertical-align: top;
            text-align: left;
        }
        th { color: var(--muted); font-weight: 600; }
        tr:last-child td { border-bottom: none; }
        .status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 11px;
            background: var(--chip);
            border: 1px solid var(--line);
        }
        .status.inflight { color: var(--warn); }
        .status.completed { color: var(--good); }
        .status.failed, .status.cancelled { color: var(--bad); }
        .mono { word-break: break-word; }
        .tiny { color: var(--muted); font-size: 11px; }
        .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
        .chip {
            padding: 6px 10px;
            border-radius: 999px;
            background: var(--chip);
            border: 1px solid var(--line);
            font-size: 12px;
            color: var(--muted);
        }
        @media (max-width: 980px) {
            .row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <h1>OnIt ROS/MCP Activity Dashboard</h1>
        <p>Live polling every second. Tracks inflight tasks, completed tasks, MCP/ROS calls, and recent LLM activity.</p>
        <div class=\"grid\" id=\"metrics\"></div>
        <div class=\"row\">
            <div class=\"table-card\">
                <div class=\"table-head\"><h2>Inflight Tasks</h2><span class=\"tiny\" id=\"updated\"></span></div>
                <div id=\"inflight-tasks\"></div>
            </div>
            <div class=\"table-card\">
                <div class=\"table-head\"><h2>Completed Tasks</h2><span class=\"tiny\">Most recent 50</span></div>
                <div id=\"completed-tasks\"></div>
            </div>
        </div>
        <div class=\"row\">
            <div class=\"table-card\">
                <div class=\"table-head\"><h2>Inflight Operations</h2><span class=\"tiny\">LLM + MCP</span></div>
                <div id=\"inflight-operations\"></div>
            </div>
            <div class=\"table-card\">
                <div class=\"table-head\"><h2>Completed Operations</h2><span class=\"tiny\">Most recent 100</span></div>
                <div id=\"completed-operations\"></div>
            </div>
        </div>
        <div class=\"row-full table-card\">
            <div class=\"table-head\"><h2>ROS/MCP Focus</h2><span class=\"tiny\">ROS-related operations surfaced from the same trace stream</span></div>
            <div id=\"ros-activity\"></div>
        </div>
    </div>
    <script>
        function esc(value) {
            if (value === null || value === undefined) return '';
            return String(value)
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;');
        }

        function statusBadge(status) {
            return `<span class=\"status ${esc(status)}\">${esc(status || 'unknown')}</span>`;
        }

        function renderMetrics(data) {
            const stats = data.stats || {};
            const metrics = [
                ['Inflight Tasks', stats.inflight_task_count ?? 0],
                ['Completed Tasks', stats.completed_task_count ?? 0],
                ['Inflight Ops', stats.inflight_operation_count ?? 0],
                ['Completed Ops', stats.completed_operation_count ?? 0],
                ['LLM Avg ms', stats.llm_avg_latency_ms ?? '-'],
                ['MCP Avg ms', stats.mcp_avg_latency_ms ?? '-'],
            ];
            document.getElementById('metrics').innerHTML = metrics.map(([label, value]) => `
                <div class=\"card\">
                    <div class=\"label\">${esc(label)}</div>
                    <div class=\"metric\">${esc(value)}</div>
                </div>
            `).join('');
        }

        function renderTable(targetId, rows, columns, emptyLabel) {
            const target = document.getElementById(targetId);
            if (!rows || rows.length === 0) {
                target.innerHTML = `<div class=\"card\"><span class=\"tiny\">${esc(emptyLabel)}</span></div>`;
                return;
            }
            const head = columns.map(col => `<th>${esc(col.label)}</th>`).join('');
            const body = rows.map(row => `<tr>${columns.map(col => `<td class=\"mono\">${col.render(row)}</td>`).join('')}</tr>`).join('');
            target.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
        }

        async function refresh() {
            const response = await fetch('/dashboard/data', { cache: 'no-store' });
            const data = await response.json();
            renderMetrics(data);
            document.getElementById('updated').textContent = `Updated ${new Date(data.generated_at).toLocaleTimeString()}`;

            renderTable('inflight-tasks', data.inflight_tasks, [
                { label: 'Status', render: row => statusBadge(row.status) },
                { label: 'Task', render: row => esc(row.task_preview) },
                { label: 'Session', render: row => esc(row.session_id) },
                { label: 'Started', render: row => `<span class=\"tiny\">${esc(row.started_at || row.timestamp)}</span>` },
            ], 'No inflight tasks.');

            renderTable('completed-tasks', data.completed_tasks, [
                { label: 'Status', render: row => statusBadge(row.status) },
                { label: 'Task', render: row => esc(row.task_preview) },
                { label: 'Result', render: row => esc(row.reason || row.response_preview || '') },
                { label: 'When', render: row => `<span class=\"tiny\">${esc(row.timestamp)}</span>` },
            ], 'No completed tasks yet.');

            renderTable('inflight-operations', data.inflight_operations, [
                { label: 'Kind', render: row => statusBadge(row.operation_kind) },
                { label: 'Name', render: row => esc(row.name) },
                { label: 'Task', render: row => esc(row.task_id) },
                { label: 'Target', render: row => esc(row.url) },
                { label: 'Started', render: row => `<span class=\"tiny\">${esc(row.started_at || row.timestamp)}</span>` },
            ], 'No inflight operations.');

            renderTable('completed-operations', data.completed_operations, [
                { label: 'Status', render: row => statusBadge(row.status) },
                { label: 'Kind', render: row => esc(row.operation_kind) },
                { label: 'Name', render: row => esc(row.name) },
                { label: 'Latency', render: row => esc(row.latency_ms ?? '-') },
                { label: 'Preview', render: row => esc(row.error || row.content_preview || '') },
            ], 'No completed operations yet.');

            renderTable('ros-activity', data.ros_activity, [
                { label: 'Status', render: row => statusBadge(row.status) },
                { label: 'Name', render: row => esc(row.name) },
                { label: 'Task', render: row => esc(row.task_id) },
                { label: 'Target', render: row => esc(row.url) },
                { label: 'Latency', render: row => esc(row.latency_ms ?? '-') },
            ], 'No ROS-specific activity detected yet.');
        }

        refresh().catch(console.error);
        setInterval(() => refresh().catch(console.error), 1000);
    </script>
</body>
</html>"""
