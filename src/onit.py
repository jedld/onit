"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OnIt: An intelligent agent framework for task automation and assistance.

"""

import asyncio
from datetime import datetime
import os
import tempfile
import yaml
import json
import uuid

from pathlib import Path
from typing import Union, Any
from pydantic import BaseModel, ConfigDict, Field
from fastmcp import Client

import logging
import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:.*")

logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from httpx/httpcore (used by FastMCP client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from openai import AsyncOpenAI

from .lib.mcp_bootstrap import ensure_mcp_servers
from .lib.tools import discover_tools
from .lib.text import remove_tags
from .lib.observability import IntrospectionServer, TraceRecorder
from .ui import ChatUI
from .model.serving.chat import chat, _resolve_api_key

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import FilePart, FileWithBytes
from a2a.utils import new_agent_text_message

AGENT_CURSOR = "OnIt"
USER_CURSOR = "You"
STOP_TAG = "<stop></stop>"


def _local_datetime_context() -> str:
    """Return a best-effort local date/time block for prompt fallback."""
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "local"
    utc_offset = now.strftime("%z")
    utc_display = f"UTC{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else "UTC"
    return (
        f"Current date: {now.strftime('%B %d, %Y')} ({now.strftime('%A')})\n"
        f"Current time: {now.strftime('%I:%M %p')} ({tz_name}, {utc_display})\n"
        f"Timezone: {tz_name}"
    )


class OnItA2AExecutor(AgentExecutor):
    """A2A executor that delegates task processing to an OnIt instance.

    Each A2A context (client conversation) gets its own isolated session
    with separate chat history, data directory, and safety queue — following
    the same pattern as the Telegram and Viber gateways.
    """

    def __init__(self, onit):
        self.onit = onit
        # Per-context session state: context_key -> {session_id, session_path, data_path, safety_queue}
        self._sessions: dict[str, dict] = {}
        # Track active safety_queue per asyncio task for disconnect middleware
        self._active_safety_queues: dict[int, asyncio.Queue] = {}

    def _get_session(self, context: 'RequestContext') -> dict:
        """Get or create session state for an A2A context."""
        # Use context_id to group related tasks from the same client,
        # fall back to task_id for one-off requests
        key = context.context_id or context.task_id or str(uuid.uuid4())
        if key not in self._sessions:
            session_id = str(uuid.uuid4())
            sessions_dir = os.path.dirname(self.onit.session_path)
            session_path = os.path.join(sessions_dir, f"{session_id}.jsonl")
            if not os.path.exists(session_path):
                with open(session_path, "w", encoding="utf-8") as f:
                    f.write("")
            data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
            os.makedirs(data_path, exist_ok=True)
            self._sessions[key] = {
                "session_id": session_id,
                "session_path": session_path,
                "data_path": data_path,
                "safety_queue": asyncio.Queue(maxsize=10),
            }
            logger.info("Created new A2A session %s for context %s", session_id, key)
        return self._sessions[key]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.get_user_input()
        if not context.message:
            raise Exception('No message provided')

        session = self._get_session(context)

        # Extract inline file parts from the A2A message and save to session data folder
        import base64
        image_paths = []
        file_paths = []
        for part in context.message.parts:
            if isinstance(part.root, FilePart) and isinstance(part.root.file, FileWithBytes):
                file_obj = part.root.file
                safe_name = os.path.basename(file_obj.name or 'file')
                filepath = os.path.join(session["data_path"], safe_name)
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(file_obj.bytes))
                if file_obj.mime_type and file_obj.mime_type.startswith('image/'):
                    image_paths.append(filepath)
                else:
                    file_paths.append(filepath)

        # Append file references to task so the agent knows about them
        if file_paths:
            file_refs = "\n".join(f"- {fp}" for fp in file_paths)
            task = f"{task}\n\nFiles uploaded to data folder:\n{file_refs}"

        # Register safety_queue for disconnect middleware
        current_task_id = id(asyncio.current_task())
        self._active_safety_queues[current_task_id] = session["safety_queue"]

        try:
            result = await self.onit.process_task(
                task,
                images=image_paths if image_paths else None,
                session_path=session["session_path"],
                data_path=session["data_path"],
                safety_queue=session["safety_queue"],
            )
        except asyncio.CancelledError:
            session["safety_queue"].put_nowait(STOP_TAG)
            raise
        finally:
            self._active_safety_queues.pop(current_task_id, None)

        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        session = self._get_session(context)
        session["safety_queue"].put_nowait(STOP_TAG)


class ClientDisconnectMiddleware:
    """ASGI middleware that signals safety_queue when a client disconnects mid-request."""

    def __init__(self, app, executor: OnItA2AExecutor):
        self.app = app
        self.executor = executor

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip disconnect detection for file upload/download routes;
        # these are normal HTTP transfers, not client task cancellations.
        path = scope.get("path", "")
        if path.startswith("/uploads"):
            await self.app(scope, receive, send)
            return

        # Read the full request body upfront
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # client already gone
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        # Provide buffered body to the inner app
        body_delivered = False
        async def buffered_receive():
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Block until cancelled (app shouldn't need receive again)
            await asyncio.Future()

        # Monitor the real receive for client disconnect
        async def disconnect_watcher():
            msg = await receive()
            if msg.get("type") == "http.disconnect":
                # Signal the safety_queue for the current request's task
                task_id = id(asyncio.current_task())
                sq = self.executor._active_safety_queues.get(task_id)
                if sq:
                    sq.put_nowait(STOP_TAG)

        watcher = asyncio.create_task(disconnect_watcher())
        try:
            await self.app(scope, buffered_receive, send)
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass


class OnIt(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    status: str = Field(default="idle")
    config_data: dict[str, Any] = Field(default_factory=dict)
    mcp_servers: list[Any] = Field(default_factory=list)
    tool_registry: Any | None = Field(default=None)
    theme: str | None = Field(default="white", exclude=True)
    messages: dict[str, str] = Field(default_factory=dict)
    stop_commands: list[str] = Field(default_factory=lambda: ['\\goodbye', '\\bye', '\\quit', '\\exit'])
    model_serving: dict[str, Any] = Field(default_factory=dict)
    user_id: str = Field(default="default_user")
    input_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    output_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    safety_queue: asyncio.Queue | None = Field(default=None, exclude=True)
    verbose: bool = Field(default=True)
    session_id: str | None = Field(default=None)
    session_path: str = Field(default="~/.onit/sessions")
    data_path: str = Field(default="")
    template_path: str | None = Field(default=None)
    documents_path: str | None = Field(default=None)
    topic: str | None = Field(default=None)
    prompt_intro: str | None = Field(default=None)
    timeout: int | None = Field(default=None)
    show_logs: bool = Field(default=False)
    loop: bool = Field(default=False)
    period: float = Field(default=10.0)
    task: str | None = Field(default=None)
    web: bool = Field(default=False)
    web_port: int = Field(default=9000)
    web_google_client_id: str | None = Field(default=None)
    web_google_client_secret: str | None = Field(default=None)
    web_allowed_emails: list[str] | None = Field(default=None)
    web_title: str = Field(default="OnIt Chat")
    a2a: bool = Field(default=False)
    a2a_port: int = Field(default=9001)
    a2a_name: str = Field(default="OnIt")
    a2a_description: str = Field(default="An intelligent agent for task automation and assistance.")
    gateway: str | None = Field(default=None)
    gateway_token: str | None = Field(default=None, exclude=True)
    viber_webhook_url: str | None = Field(default=None)
    viber_port: int = Field(default=8443)
    prompt_url: str | None = Field(default=None, exclude=True)
    file_server_url: str | None = Field(default=None, exclude=True)
    chat_ui: Any | None = Field(default=None, exclude=True)
    auto_summarize: bool = Field(default=True)
    auto_summarize_threshold: int = Field(default=52000)
    auto_summarize_keep_recent: int = Field(default=6)
    session_history_max_turns: int | None = Field(default=None)
    session_history_token_budget: int | None = Field(default=None)
    context_window: int = Field(default=262144)
    trace_recorder: Any | None = Field(default=None, exclude=True)
    introspection_server: Any | None = Field(default=None, exclude=True)
    trace_enabled: bool = Field(default=True)
    trace_dir: str = Field(default="~/.onit/introspection")
    trace_max_events: int = Field(default=2000)
    introspection_enabled: bool = Field(default=True)
    introspection_host: str = Field(default="127.0.0.1")
    introspection_port: int = Field(default=9100)

    def __init__(self, config: Union[str, os.PathLike[str], dict[str, Any], None] = None) -> None :
        super().__init__()

        if config is not None:
            if isinstance(config, (str, os.PathLike)):
                cfg_path = Path(config).expanduser()
                if not cfg_path.exists():
                    raise FileNotFoundError(f"Config file {cfg_path} not found.")
                with cfg_path.open("r", encoding="utf-8") as f:
                    self.config_data = yaml.safe_load(f) or {}
            elif isinstance(config, dict):
                self.config_data = config
            else:
                raise TypeError("config must be a path-like object or dict.")

        self.initialize()
        if not self.loop:
            if self.web:
                from .ui.web import WebChatUI
                self.chat_ui = WebChatUI(
                    theme=self.theme,
                    data_path=self.data_path,
                    show_logs=self.show_logs,
                    server_port=self.web_port,
                    google_client_id=self.web_google_client_id,
                    google_client_secret=self.web_google_client_secret,
                    allowed_emails=self.web_allowed_emails,
                    session_path=self.session_path,
                    title=self.web_title,
                    verbose=self.verbose,
                )
                self.chat_ui._onit = self
            else:
                if self.a2a:
                    banner = "OnIt Agent to Agent Server"
                elif self.gateway:
                    banner = f"OnIt {self.gateway.capitalize()} Gateway"
                else:
                    banner = "OnIt Chat Interface"
                self.chat_ui = ChatUI(self.theme, show_logs=self.show_logs, banner_title=banner)
        
    def initialize(self):
        observability = self.config_data.get('observability', {})
        self.trace_enabled = observability.get('tracing', True)
        self.trace_dir = os.path.expanduser(observability.get('trace_dir', '~/.onit/introspection'))
        self.trace_max_events = int(observability.get('max_events', 2000))
        self.introspection_enabled = observability.get('introspection', True)
        self.introspection_host = observability.get('host', '127.0.0.1')
        self.introspection_port = int(observability.get('port', 9100))
        self.trace_recorder = TraceRecorder(
            base_dir=self.trace_dir,
            enabled=self.trace_enabled,
            max_events=self.trace_max_events,
        )
        self.mcp_servers = self.config_data['mcp']['servers'] if 'mcp' in self.config_data and 'servers' in self.config_data['mcp'] else []
        # Override MCP server URL hosts if mcp_host is configured
        mcp_host = self.config_data.get('mcp', {}).get('mcp_host')
        if mcp_host:
            from urllib.parse import urlparse, urlunparse
            for server in self.mcp_servers:
                url = server.get('url')
                if url:
                    parsed = urlparse(url)
                    server['url'] = urlunparse(parsed._replace(netloc=f"{mcp_host}:{parsed.port}" if parsed.port else mcp_host))
        ensure_mcp_servers(
            self.config_data,
            log_level='DEBUG' if self.config_data.get('verbose') else 'ERROR',
        )
        # Find the prompts server URL from the MCP servers list
        for server in self.mcp_servers:
            if server.get('name') == 'PromptsMCPServer' and server.get('enabled', True):
                self.prompt_url = server.get('url')
                break
        if not self.prompt_url:
            raise ValueError(
                "PromptsMCPServer not found or disabled in MCP server config. "
                "Ensure it is listed under mcp.servers with a valid URL."
            )
        tool_servers = [s for s in self.mcp_servers if s.get('name') != 'PromptsMCPServer']
        a2a_agents = self.config_data.get('a2a_agents', [])
        # If the main model supports vision, skip A2A agents that are pure VLM
        # delegates with no other value (marked with vision_only: true in the config).
        # Agents with vision_only: false (or unset) are kept even when the main model
        # supports vision because they may have additional strengths (e.g. robotics,
        # domain expertise).
        main_model_vision = self.config_data.get('serving', {}).get('vision', False)
        if main_model_vision:
            skipped = [a.get('name', '?') for a in a2a_agents if a.get('vision_only', False)]
            if skipped:
                print(f"  [vision] Main model supports vision — skipping vision-only agent(s): {', '.join(skipped)}")
            a2a_agents = [a for a in a2a_agents if not a.get('vision_only', False)]
        self.tool_registry = asyncio.run(discover_tools(tool_servers, a2a_agents=a2a_agents))
        self.tool_registry.set_observer(self.trace_recorder)
        # List discovered tools
        for tool_name in self.tool_registry:
            print(f"  - {tool_name}")
        print(f"  Total: {len(self.tool_registry)} tools discovered")
        self.theme = self.config_data.get('theme', 'white')
        self.messages = self.config_data.get('messages', {})
        self.stop_commands = list(self.config_data.get('stop_command', self.stop_commands))
        self.model_serving = self.config_data.get('serving', {})
        # resolve host: CLI/config > env var ONIT_HOST
        if 'host' not in self.model_serving or not self.model_serving['host']:
            env_host = os.environ.get('ONIT_HOST')
            if env_host:
                self.model_serving['host'] = env_host
            else:
                raise ValueError(
                    "No serving host configured. Set it via:\n"
                    "  - ONIT_HOST environment variable\n"
                    "  - --host CLI flag\n"
                    "  - serving.host in the config YAML"
                )
                self.context_window = int(self.model_serving.get('context_window', 262144))
        self.user_id = self.config_data.get('user_id', 'default_user')
        self.status = "initialized"
        self.verbose = self.config_data.get('verbose', False)
        # Suppress noisy logs unless verbose
        if not self.verbose:
            logging.getLogger("src.lib.tools").setLevel(logging.WARNING)
            logging.getLogger("lib.tools").setLevel(logging.WARNING)
            logging.getLogger("type.tools").setLevel(logging.WARNING)
            logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
            logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        # append session id to sessions path
        self.session_id = str(uuid.uuid4())
        self.session_path = os.path.join(self.config_data.get('session_path', '~/.onit/sessions'), f"{self.session_id}.jsonl")
        # create the sessions directory and file if not exists. expand ~ to home directory
        self.session_path = os.path.expanduser(self.session_path)
        sessions_dir = os.path.dirname(self.session_path)
        os.makedirs(sessions_dir, exist_ok=True)
        if not os.path.exists(self.session_path):
            with open(self.session_path, "w", encoding="utf-8") as f:
                f.write("")
        self.data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / self.session_id)
        os.makedirs(self.data_path, exist_ok=True)
        # Compute file_server_url for file transfer via callback_url
        self.file_server_url = None
        mcp_host = self.config_data.get('mcp', {}).get('mcp_host')
        if mcp_host and self.config_data.get('web', False):
            import socket
            web_port = self.config_data.get('web_port', 9000)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((mcp_host, 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "127.0.0.1"
            self.file_server_url = f"http://{local_ip}:{web_port}"
        elif self.config_data.get('a2a', False):
            # In A2A mode, serve files through the A2A server itself
            import socket
            a2a_port = self.config_data.get('a2a_port', 9001)
            if mcp_host:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect((mcp_host, 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                except Exception:
                    local_ip = "127.0.0.1"
            else:
                local_ip = "127.0.0.1"
            self.file_server_url = f"http://{local_ip}:{a2a_port}"
        self.template_path = self.config_data.get('template_path', None)
        self.documents_path = self.config_data.get('documents_path', None)
        self.topic = self.config_data.get('topic', None)
        self.prompt_intro = self.config_data.get('prompt_intro', None)
        self.timeout = self.config_data.get('timeout', None)  # default timeout 300 seconds
        if self.timeout is not None and self.timeout < 0:
            self.timeout = None  # no timeout
        self.show_logs = self.config_data.get('show_logs', False)
        self.loop = self.config_data.get('loop', False)
        self.period = float(self.config_data.get('period', 20.0))
        self.task = self.config_data.get('task', None)
        self.web = self.config_data.get('web', False)
        self.web_port = self.config_data.get('web_port', 9000)
        self.web_google_client_id = self.config_data.get('web_google_client_id', None)
        self.web_google_client_secret = self.config_data.get('web_google_client_secret', None)
        # Nullify placeholder credentials so auth is cleanly disabled
        for attr in ('web_google_client_id', 'web_google_client_secret'):
            val = getattr(self, attr, None)
            if val and "YOUR_" in str(val).upper():
                setattr(self, attr, None)
        self.web_allowed_emails = self.config_data.get('web_allowed_emails', None)
        self.web_title = self.config_data.get('web_title', 'OnIt Chat')
        self.a2a = self.config_data.get('a2a', False)
        self.a2a_port = self.config_data.get('a2a_port', 9001)
        self.a2a_name = self.config_data.get('a2a_name', 'OnIt')
        self.a2a_description = self.config_data.get('a2a_description', 'An intelligent agent for task automation and assistance.')
        self.gateway = self.config_data.get('gateway', None) or None
        self.gateway_token = self.config_data.get('gateway_token', None)
        self.viber_webhook_url = self.config_data.get('viber_webhook_url', None)
        self.viber_port = self.config_data.get('viber_port', 8443)
        self.auto_summarize = self.config_data.get('auto_summarize', True)
        configured_threshold = self.config_data.get('auto_summarize_threshold', None)
        if configured_threshold is None:
            self.auto_summarize_threshold = self._default_auto_summarize_threshold()
        else:
            configured_threshold = int(configured_threshold)
            if configured_threshold == 52000 and self.context_window <= 32768:
                self.auto_summarize_threshold = self._default_auto_summarize_threshold()
            else:
                self.auto_summarize_threshold = configured_threshold
        self.auto_summarize_keep_recent = int(self.config_data.get('auto_summarize_keep_recent', 6))
        configured_history_turns = self.config_data.get('session_history_max_turns', None)
        if configured_history_turns is None:
            self.session_history_max_turns = self._default_session_history_max_turns()
        else:
            self.session_history_max_turns = int(configured_history_turns)
        configured_history_token_budget = self.config_data.get('session_history_token_budget', None)
        if configured_history_token_budget is None:
            self.session_history_token_budget = self._default_session_history_token_budget()
        else:
            self.session_history_token_budget = int(configured_history_token_budget)

    def _ensure_introspection_server(self) -> None:
        if not self.introspection_enabled or not self.trace_recorder:
            return
        if self.introspection_server is not None:
            return
        self.introspection_server = IntrospectionServer(
            trace_recorder=self.trace_recorder,
            onit_ref=self,
            host=self.introspection_host,
            port=self.introspection_port,
        )
        self.introspection_server.start()

    def get_probe_report(self) -> dict[str, Any]:
        report = self.trace_recorder.build_probe_report(
            model_host=self.model_serving.get('host'),
            mcp_servers=self.mcp_servers,
            tool_registry=self.tool_registry,
        )
        data = report.to_dict()
        data['trace'] = self.trace_recorder.stats_snapshot()
        data['session_path'] = self.session_path
        data['introspection'] = {
            'enabled': self.introspection_enabled,
            'host': self.introspection_host,
            'port': self.introspection_port,
        }
        return data

    @staticmethod
    def _derive_session_id(session_path: str | None, fallback: str | None = None) -> str | None:
        if session_path:
            return Path(session_path).stem
        return fallback
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 characters."""
        return max(1, len(text) // 4)

    def _default_auto_summarize_threshold(self) -> int:
        """Pick a summary threshold that scales with the model context window."""
        return max(2048, int(self.context_window * 0.25))

    def _default_session_history_max_turns(self) -> int:
        """Keep fewer verbatim turns for smaller context windows."""
        if self.context_window <= 16384:
            return 4
        if self.context_window <= 32768:
            return 8
        if self.context_window <= 65536:
            return 12
        return 20

    def _default_session_history_token_budget(self) -> int:
        """Reserve only a modest portion of the context window for raw history."""
        return max(1024, int(self.context_window * 0.15))

    async def _auto_summarize_history(self, session_path: str) -> None:
        """If session history exceeds the token threshold, summarize old turns via the LLM
        and rewrite the session file with a compact summary entry + the most recent turns."""
        # Load all raw entries
        all_entries = []
        try:
            if not os.path.exists(session_path):
                return
            with open(session_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            all_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return

        if not all_entries:
            return

        # Separate an existing leading summary from normal turns
        existing_summary = None
        normal_entries = []
        for entry in all_entries:
            if entry.get("type") == "summary":
                existing_summary = entry  # keep the last summary seen as the base
            else:
                normal_entries.append(entry)

        # Estimate total tokens in history
        total_tokens = 0
        if existing_summary:
            total_tokens += self._estimate_tokens(existing_summary.get("content", ""))
        for e in normal_entries:
            total_tokens += self._estimate_tokens(e.get("task", "") + e.get("response", ""))

        if total_tokens < self.auto_summarize_threshold:
            return  # Nothing to do

        keep = self.auto_summarize_keep_recent
        turns_to_summarize = normal_entries[:-keep] if len(normal_entries) > keep else []
        recent_turns = normal_entries[-keep:] if len(normal_entries) >= keep else normal_entries

        if not turns_to_summarize and not existing_summary:
            return  # Not enough old turns to summarize

        # Build the text to summarize
        lines = []
        if existing_summary:
            lines.append(f"[Prior summary]: {existing_summary['content']}")
        for e in turns_to_summarize:
            lines.append(f"User: {e['task']}")
            lines.append(f"Assistant: {e['response']}")
        history_text = "\n".join(lines)

        summary_prompt = (
            "Summarize the following conversation history concisely, preserving key facts, "
            "decisions, file paths, and any important context the assistant may need later. "
            "Be brief but complete.\n\n" + history_text
        )

        try:
            host = self.model_serving["host"]
            model = self.model_serving["model"]
            host_key = self.model_serving.get("host_key", "EMPTY")
            api_key = _resolve_api_key(host, host_key)
            aclient = AsyncOpenAI(base_url=host, api_key=api_key)
            completion = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise summarization assistant."},
                    {"role": "user", "content": summary_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            summary_text = completion.choices[0].message.content.strip()
            if "</think>" in summary_text:
                summary_text = summary_text.split("</think>")[-1].strip()
        except Exception as e:
            logger.warning("Auto-summarize LLM call failed: %s", e)
            return

        # Rewrite session file: [summary entry, ...recent turns]
        new_summary_entry = {
            "type": "summary",
            "content": summary_text,
            "turns_covered": (len(turns_to_summarize) + (1 if existing_summary else 0)),
            "timestamp": asyncio.get_event_loop().time(),
        }
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(new_summary_entry) + "\n")
                for e in recent_turns:
                    f.write(json.dumps(e) + "\n")
            logger.info("Auto-summarized session: compressed %d turns into summary, kept %d recent.",
                        len(turns_to_summarize), len(recent_turns))
        except Exception as e:
            logger.warning("Failed to write summarized session file: %s", e)

    def load_session_history(self, max_turns: int | None = None, session_path: str | None = None) -> list[dict]:
        """Load recent session history from the JSONL session file.

        Args:
            max_turns: Maximum number of recent task/response pairs to return.
            session_path: Optional override path to the session file.

        Returns:
            A list of dicts with 'task' and 'response' keys, oldest first.
        """
        effective_path = session_path or self.session_path
        history = []
        try:
            if os.path.exists(effective_path):
                with open(effective_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                if entry.get("type") == "summary":
                                    # Convert summary to a pseudo-turn so chat.py sees it as context
                                    history.append({
                                        "task": "[Previous conversation summary]",
                                        "response": entry["content"],
                                    })
                                elif "task" in entry and "response" in entry:
                                    history.append(entry)
                            except json.JSONDecodeError:
                                continue
        except Exception:
            pass
        if max_turns is None:
            max_turns = self.session_history_max_turns or self._default_session_history_max_turns()
        if max_turns <= 0:
            return []
        history = history[-max_turns:]
        token_budget = self.session_history_token_budget or self._default_session_history_token_budget()
        if token_budget > 0:
            kept_history = []
            tokens_used = 0
            for entry in reversed(history):
                entry_tokens = self._estimate_tokens(entry.get("task", "") + entry.get("response", ""))
                if kept_history and tokens_used + entry_tokens > token_budget:
                    break
                kept_history.append(entry)
                tokens_used += entry_tokens
            history = list(reversed(kept_history))
        # return only the most recent turns
        return history

    async def _build_instruction(self,
                                 task: str,
                                 data_path: str,
                                 session_id: str | None,
                                 task_id: str | None = None) -> str:
        start_time = asyncio.get_event_loop().time()
        operation_id = f"mcp_{uuid.uuid4().hex[:12]}"
        prompt_args = {
            "task": task,
            "data_path": data_path,
            "template_path": self.template_path,
            "file_server_url": self.file_server_url,
            "documents_path": self.documents_path,
            "topic": self.topic,
        }
        if self.trace_recorder:
            self.trace_recorder.record(
                'mcp.request',
                session_id=session_id,
                task_id=task_id,
                operation_id=operation_id,
                tool_name='assistant_prompt',
                url=self.prompt_url,
                arguments=prompt_args,
                kind='prompt',
            )

        prompt_client = Client(self.prompt_url)
        try:
            async with prompt_client:
                instruction = await prompt_client.get_prompt("assistant", prompt_args)
                text = instruction.messages[0].content.text
            if self.trace_recorder:
                self.trace_recorder.record(
                    'mcp.response',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name='assistant_prompt',
                    url=self.prompt_url,
                    latency_ms=round((asyncio.get_event_loop().time() - start_time) * 1000, 2),
                    result_type='prompt',
                    result_preview=text,
                    kind='prompt',
                )
            return text
        except Exception as exc:
            if self.trace_recorder:
                self.trace_recorder.record(
                    'mcp.error',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name='assistant_prompt',
                    url=self.prompt_url,
                    latency_ms=round((asyncio.get_event_loop().time() - start_time) * 1000, 2),
                    error=str(exc),
                    kind='prompt',
                )
            logger.warning(
                "Prompts MCP unavailable at %s; falling back to local instruction builder: %s",
                self.prompt_url,
                exc,
            )
            text = self._build_local_instruction(task=task, data_path=data_path)
            if self.trace_recorder:
                self.trace_recorder.record(
                    'mcp.response',
                    session_id=session_id,
                    task_id=task_id,
                    operation_id=operation_id,
                    tool_name='assistant_prompt',
                    url=self.prompt_url,
                    latency_ms=round((asyncio.get_event_loop().time() - start_time) * 1000, 2),
                    result_type='prompt_fallback',
                    result_preview=text,
                    kind='prompt',
                    fallback=True,
                    fallback_reason=str(exc),
                )
            return text

    def _build_local_instruction(self, task: str, data_path: str) -> str:
        """Build a local prompt when the prompt MCP server is unavailable."""
        Path(data_path).mkdir(parents=True, exist_ok=True)
        current_date = datetime.now().strftime("%B %d, %Y")
        default_template = (
            "You are an autonomous agent with access to tools and a file system.\n\n"
            "## Context\n"
            f"- **Today's date**: {current_date}\n"
            f"- **Working directory**: {data_path} — sandbox folder for reading and writing files.\n\n"
            "## Constraints\n"
            f"- NEVER create or modify files outside of `{data_path}`.\n\n"
            "## Task\n"
            f"{task}\n\n"
            "## Execution Policy\n"
            "- Translate the task into observable success conditions before acting.\n"
            "- Use tools to gather evidence, act in short verifiable steps, and re-check after each material action.\n"
            "- Do not claim success from stale, partial, or ambiguous observations.\n\n"
            "## Persistent Landmark Hypothesis Contract\n"
            "When the task involves a physical object, place, doorway, person, landmark, or requested viewing relation:\n"
            "- After the first credible detection, create and preserve a landmark hypothesis instead of restarting the search from scratch on every new frame.\n"
            "- Maintain and update these structured fields whenever evidence improves:\n"
            "  - `object_label`\n"
            "  - `estimated_bearing_deg`\n"
            "  - `approximate_range_m`\n"
            "  - `visible_face`\n"
            "  - `last_confirmed_pose`\n"
            "  - `confidence`\n"
            "- Reuse the current landmark hypothesis after each camera refresh or motion step; update it incrementally rather than discarding it.\n"
            "- If a fresh observation is weak or ambiguous, treat that as a local reacquisition problem around the existing hypothesis, not as a reason to restart global search.\n\n"
            "## Relational Task Decomposition\n"
            "For tasks such as going behind, around, beside, or otherwise achieving a spatial relationship to a target, you MUST follow this order:\n"
            "1. Confirm the target identity with direct evidence.\n"
            "2. Determine which face or side of the target is currently visible.\n"
            "3. Choose a left or right circumnavigation direction based on clearance and safety.\n"
            "4. Move in short segments while preserving the landmark hypothesis.\n"
            "5. Re-verify target identity and visible face after every short segment.\n"
            "6. Verify the requested final relation, such as rear-side evidence, before claiming success.\n\n"
            "## Loss Handling\n"
            "- If the target is lost, first attempt bounded local reacquisition using the current landmark hypothesis.\n"
            "- Abort the relational maneuver and report the blocker if the target remains unconfirmed for 3 consecutive local verification steps, or if the path is blocked or unsafe.\n"
        )
        template = default_template

        if self.template_path:
            template_file = Path(self.template_path)
            if template_file.exists() and template_file.suffix in ('.yaml', '.yml'):
                with open(template_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                template = config.get('instruction_template', default_template)

        instruction = (
            "## Current Date & Time\n"
            f"{_local_datetime_context()}\n\n"
            + template.format(
                task=task,
                current_date=current_date,
                data_path=data_path,
                session_id=Path(data_path).name,
            )
        )

        if self.topic and self.topic != "null":
            instruction += (
                "\n## Topic\n"
                f"Unless specified, assume that the topic is about `{self.topic}`.\n"
            )

        if self.file_server_url and self.file_server_url != "null":
            upload_id = Path(data_path).name
            upload_prefix = f"{self.file_server_url}/uploads/{upload_id}"
            instruction += (
                f"\nFiles are served by a remote file server at {upload_prefix}/.\n"
                "Before reading any file referenced in the task, first download it:\n"
                f"  curl -s {upload_prefix}/<filename> -o {data_path}/<filename>\n"
                "After creating or saving any output file, upload it back to the file server:\n"
                f"  curl -s -X POST -F 'file=@{data_path}/<filename>' {upload_prefix}/\n"
                "Always download before reading and upload after writing.\n"
                "When using create_presentation, create_excel, or create_document tools, always pass "
                f"callback_url=\"{upload_prefix}\" so files are automatically uploaded.\n"
            )

        if self.documents_path and self.documents_path != "null":
            instruction += (
                "\n## Relevant Information\n"
                f"Search and read related documents (PDF, TXT, DOCX, XLSX, PPTX, and Markdown (MD)) in `{self.documents_path}`.\n"
                "Search the web for additional information if and only if above documents are insufficient to complete the task.\n"
            )

        instruction += (
            "\n## Instructions\n"
            "1. If the answer is straightforward, respond directly without tool use.\n"
            "2. Otherwise, reason step by step, invoke tools as needed, and work toward a final answer.\n"
            "3. If critical information is missing and cannot be inferred, ask exactly one clarifying question before proceeding.\n"
            "4. If a file was generated, provide a download link to the file.\n"
            "5. Conclude with your final answer in this format:\n\n"
            "<your answer here>\n"
        )
        return instruction

    def _history_token_estimate(self, history: list[dict]) -> int:
        """Estimate total tokens in a loaded session history list."""
        total = 0
        for e in history:
            total += self._estimate_tokens(e.get("task", "") + e.get("response", ""))
        return total

    async def run(self) -> None:
        """Run the OnIt agent session"""
        try:
            self._ensure_introspection_server()
            self.input_queue = asyncio.Queue(maxsize=10)
            self.output_queue = asyncio.Queue(maxsize=10)
            self.safety_queue = asyncio.Queue(maxsize=10)
            # safety_queue is used by non-web modes; web uses per-session queues
            self.status = "running"
            if self.a2a:
                await self.run_a2a()
            elif self.loop:
                await self.run_loop()
            else:
                if self.web and hasattr(self.chat_ui, 'launch'):
                    self.chat_ui.launch(asyncio.get_event_loop())
                    # Web sessions call process_task() directly; keep loop alive
                    while self.status == "running":
                        await asyncio.sleep(1)
                else:
                    client_to_agent_task = asyncio.create_task(self.client_to_agent())
                    await asyncio.gather(client_to_agent_task)
        except Exception:
            pass
        finally:
            self.status = "stopped"

    async def process_task(self, task: str, images: list[str] | None = None,
                           session_path: str | None = None,
                           data_path: str | None = None,
                           safety_queue: asyncio.Queue | None = None,
                           task_id: str | None = None) -> str:
        """Process a single task and return the response string.

        Args:
            task: The user task/message to process.
            images: Optional list of image file paths.
            session_path: Optional override for session history file path.
            data_path: Optional override for data directory path.
            safety_queue: Optional per-session safety queue (e.g. per-tab in web UI).
        """
        # Use per-chat overrides if provided, otherwise fall back to instance defaults
        effective_session_path = session_path or self.session_path
        effective_data_path = data_path or self.data_path
        effective_safety_queue = safety_queue or self.safety_queue
        if effective_safety_queue is None:
            effective_safety_queue = asyncio.Queue(maxsize=10)
        effective_session_id = self._derive_session_id(effective_session_path, self.session_id)
        effective_task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"

        if self.trace_recorder:
            self.trace_recorder.record(
                'agent.task_received',
                session_id=effective_session_id,
            task_id=effective_task_id,
                task_preview=task,
                image_count=len(images or []),
                session_path=effective_session_path,
                data_path=effective_data_path,
            )

        while not effective_safety_queue.empty():
            effective_safety_queue.get_nowait()

        instruction = await self._build_instruction(
            task=task,
            data_path=effective_data_path,
            session_id=effective_session_id,
            task_id=effective_task_id,
        )

        kwargs = {
            'console': None, 'chat_ui': None,
            'cursor': AGENT_CURSOR, 'memories': None,
            'verbose': self.verbose,
            'data_path': effective_data_path,
            'max_tokens': self.model_serving.get('max_tokens', 262144),
            'session_history': self.load_session_history(session_path=effective_session_path),
            'trace_recorder': self.trace_recorder,
            'session_id': effective_session_id,
            'task_id': effective_task_id,
        }
        if self.prompt_intro:
            kwargs['prompt_intro'] = self.prompt_intro
        last_response = await chat(
            host=self.model_serving["host"],
            host_key=self.model_serving.get("host_key", "EMPTY"),
            model=self.model_serving["model"],
            instruction=instruction,
            images=images,
            tool_registry=self.tool_registry,
            safety_queue=effective_safety_queue,
            think=self.model_serving["think"],
            timeout=self.timeout,
            **kwargs,
        )

        if last_response is None:
            logger.error("chat() returned None — likely a safety queue trigger or unhandled error. "
                         "Host: %s, Model: %s", self.model_serving["host"], self.model_serving["model"])
            if self.trace_recorder:
                self.trace_recorder.record(
                    'agent.task_failed',
                    session_id=effective_session_id,
                    task_id=effective_task_id,
                    task_preview=task,
                    reason='chat returned none',
                )
            return "I am sorry \U0001f614. Could you please rephrase your question?"

        response = remove_tags(last_response)
        if self.trace_recorder:
            self.trace_recorder.record(
                'agent.task_completed',
                session_id=effective_session_id,
                task_id=effective_task_id,
                task_preview=task,
                response_preview=response,
                response_length=len(response),
            )
        try:
            with open(effective_session_path, "a", encoding="utf-8") as f:
                session_data = {
                    "task": task,
                    "response": response,
                    "timestamp": asyncio.get_event_loop().time(),
                }
                f.write(json.dumps(session_data) + "\n")
        except Exception:
            pass

        # Auto-summarize if history has grown too large
        if self.auto_summarize:
            try:
                await self._auto_summarize_history(effective_session_path)
            except Exception as e:
                logger.warning("Auto-summarize failed: %s", e)

        return response

    async def run_loop(self) -> None:
        """Run the OnIt agent in loop mode, executing a task repeatedly."""
        if not self.task:
            raise ValueError("Loop mode requires a 'task' to be set in the config.")

        print(f"Loop mode: task='{self.task}', period={self.period}s (Ctrl+C to stop)")
        iteration = 0

        while True:
            try:
                iteration += 1
                start_time = asyncio.get_event_loop().time()
                task_id = f"task_{uuid.uuid4().hex[:12]}"

                # clear safety queue
                while not self.safety_queue.empty():
                    self.safety_queue.get_nowait()

                if self.trace_recorder:
                    self.trace_recorder.record(
                        'agent.task_received',
                        session_id=self.session_id,
                        task_id=task_id,
                        task_preview=self.task,
                        image_count=0,
                        session_path=self.session_path,
                        data_path=self.data_path,
                    )

                # build instruction via MCP prompt
                print(f"--- Iteration {iteration} ---")
                instruction = await self._build_instruction(
                    task=self.task,
                    data_path=self.data_path,
                    session_id=self.session_id,
                    task_id=task_id,
                )

                # call chat directly (no queues needed)
                kwargs = {'console': None,
                          'chat_ui': None,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'verbose': self.verbose,
                          'data_path': self.data_path,
                          'max_tokens': self.model_serving.get('max_tokens', 262144),
                          'session_history': self.load_session_history(),
                          'trace_recorder': self.trace_recorder,
                          'session_id': self.session_id,
                          'task_id': task_id}
                last_response = await chat(host=self.model_serving["host"],
                                            host_key=self.model_serving.get("host_key", "EMPTY"),
                                            model=self.model_serving["model"],
                                            instruction=instruction,
                                            tool_registry=self.tool_registry,
                                            safety_queue=self.safety_queue,
                                            think=self.model_serving["think"],
                                            timeout=self.timeout,
                                            **kwargs)

                if last_response is not None:
                    elapsed_time = asyncio.get_event_loop().time() - start_time
                    response = remove_tags(last_response)
                    if self.trace_recorder:
                        self.trace_recorder.record(
                            'agent.task_completed',
                            session_id=self.session_id,
                            task_id=task_id,
                            task_preview=self.task,
                            response_preview=response,
                            response_length=len(response),
                        )
                    print(f"\n[{AGENT_CURSOR}] ({elapsed_time:.2f}s)\n{response}\n")

                    # save to session JSONL
                    try:
                        with open(self.session_path, "a", encoding="utf-8") as f:
                            session_data = {
                                "task": self.task,
                                "response": response,
                                "timestamp": asyncio.get_event_loop().time()
                            }
                            f.write(json.dumps(session_data) + "\n")
                    except Exception:
                        pass

                # countdown timer before next iteration
                remaining = int(self.period)
                while remaining > 0:
                    print(f"\rNext in {remaining}s (Ctrl+C to stop)  ", end="", flush=True)
                    await asyncio.sleep(1)
                    remaining -= 1
                # sleep any fractional remainder
                frac = self.period - int(self.period)
                if frac > 0:
                    await asyncio.sleep(frac)
                print("\r" + " " * 40 + "\r", end="", flush=True)

            except asyncio.CancelledError:
                if self.trace_recorder:
                    self.trace_recorder.record(
                        'agent.task_cancelled',
                        session_id=self.session_id,
                        task_id=task_id if 'task_id' in locals() else None,
                        task_preview=self.task,
                        reason='loop cancelled',
                    )
                return
            except KeyboardInterrupt:
                if self.trace_recorder:
                    self.trace_recorder.record(
                        'agent.task_cancelled',
                        session_id=self.session_id,
                        task_id=task_id if 'task_id' in locals() else None,
                        task_preview=self.task,
                        reason='keyboard interrupt',
                    )
                return
            except Exception:
                if self.trace_recorder:
                    self.trace_recorder.record(
                        'agent.task_failed',
                        session_id=self.session_id,
                        task_id=task_id if 'task_id' in locals() else None,
                        task_preview=self.task,
                        reason='loop iteration failed',
                    )
                await asyncio.sleep(self.period)

    async def run_a2a(self) -> None:
        """Run OnIt as an A2A server, accepting tasks from other agents."""
        import uvicorn
        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore
        from a2a.server.events import InMemoryQueueManager
        from a2a.types import AgentCard, AgentCapabilities, AgentSkill

        agent_card = AgentCard(
            name=self.a2a_name,
            description=self.a2a_description,
            url=f"http://0.0.0.0:{self.a2a_port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=False),
            skills=[AgentSkill(
                id="general",
                name="General Task",
                description="Process any task using OnIt's tools and LLM capabilities.",
                tags=["general", "automation"],
            )],
        )

        executor = OnItA2AExecutor(self)
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            queue_manager=InMemoryQueueManager(),
        )
        a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        starlette_app = a2a_app.build()

        # Add file upload/download routes so MCP tools can send files
        # back through the A2A server instead of requiring a separate file server
        from starlette.requests import Request
        from starlette.responses import FileResponse, Response, JSONResponse
        from starlette.routing import Route

        def _find_session_data_path(session_id: str) -> str | None:
            """Look up per-context data_path by session_id."""
            for session in executor._sessions.values():
                if session["session_id"] == session_id:
                    return session["data_path"]
            return None

        async def serve_upload(request: Request) -> Response:
            session_id = request.path_params["session_id"]
            session_data_path = _find_session_data_path(session_id)
            if session_data_path is None:
                return Response(content="Session not found", status_code=404)
            filename = request.path_params["filename"]
            safe_name = os.path.basename(filename)
            filepath = os.path.join(session_data_path, safe_name)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "rb") as f:
                        content = f.read()
                    import mimetypes
                    media_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
                    return Response(content=content, media_type=media_type)
                except OSError:
                    return Response(content="File read error", status_code=500)
            return Response(content="File not found", status_code=404)

        async def receive_upload(request: Request) -> Response:
            session_id = request.path_params["session_id"]
            session_data_path = _find_session_data_path(session_id)
            if session_data_path is None:
                return Response(content="Session not found", status_code=404)
            from starlette.formparsers import MultiPartParser
            os.makedirs(session_data_path, exist_ok=True)
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                return JSONResponse({"error": "No file provided"}, status_code=400)
            safe_name = os.path.basename(upload.filename)
            filepath = os.path.join(session_data_path, safe_name)
            content = await upload.read()
            with open(filepath, "wb") as f:
                f.write(content)
            await form.close()
            return JSONResponse({"filename": safe_name, "status": "ok"})

        starlette_app.routes.insert(0, Route("/uploads/{session_id}/{filename}", serve_upload, methods=["GET"]))
        starlette_app.routes.insert(0, Route("/uploads/{session_id}/", receive_upload, methods=["POST"]))

        # Wrap app with disconnect detection middleware
        wrapped_app = ClientDisconnectMiddleware(starlette_app, executor)

        print(f"A2A server running at http://0.0.0.0:{self.a2a_port}/ (Ctrl+C to stop)")

        config = uvicorn.Config(wrapped_app, host="0.0.0.0", port=self.a2a_port, log_level="info" if self.verbose else "warning", access_log=self.verbose)
        server = uvicorn.Server(config)
        await server.serve()

    def run_gateway_sync(self) -> None:
        """Run OnIt as a messaging gateway (blocking, owns the event loop).

        Supports Telegram and Viber gateways based on ``self.gateway`` value.
        """
        self.input_queue = asyncio.Queue(maxsize=10)
        self.output_queue = asyncio.Queue(maxsize=10)
        self.safety_queue = asyncio.Queue(maxsize=10)
        self.status = "running"
        self._ensure_introspection_server()

        if self.gateway == "viber":
            from .ui.viber import ViberGateway

            if not self.gateway_token:
                raise ValueError(
                    "Viber gateway requires a bot token. Set VIBER_BOT_TOKEN "
                    "environment variable or gateway_token in config."
                )
            if not self.viber_webhook_url:
                raise ValueError(
                    "Viber gateway requires a webhook URL. Set VIBER_WEBHOOK_URL "
                    "environment variable or --viber-webhook-url CLI option."
                )
            gw = ViberGateway(
                self, self.gateway_token,
                webhook_url=self.viber_webhook_url,
                port=self.viber_port,
                show_logs=self.show_logs,
            )
        else:
            from .ui.telegram import TelegramGateway

            if not self.gateway_token:
                raise ValueError(
                    "Telegram gateway requires a bot token. Set TELEGRAM_BOT_TOKEN "
                    "environment variable or gateway_token in config."
                )
            gw = TelegramGateway(self, self.gateway_token, show_logs=self.show_logs)

        gw.run_sync()

    async def client_to_agent(self) -> None:
        """Handle client to agent communication"""

        agent_task = None
        loop = asyncio.get_event_loop()
        safety_warning = self.messages.get('safety_warning', "Press 'Enter' key to stop all tasks.")

        while True:
            if self.web:
                task = await self.chat_ui.get_user_input_async()
            else:
                task = await loop.run_in_executor(None, self.chat_ui.get_user_input)

            if task.lower().strip() in self.stop_commands:
                if not self.web:
                    self.chat_ui.console.print("Exiting chat session...", style="warning")
                if agent_task and not agent_task.done():
                    agent_task.cancel()
                break
            if not task or len(task) == 0:
                task = None
                continue

            task_id = f"task_{uuid.uuid4().hex[:12]}"
            if self.trace_recorder:
                self.trace_recorder.record(
                    'agent.task_received',
                    session_id=self.session_id,
                    task_id=task_id,
                    task_preview=task,
                    image_count=0,
                    session_path=self.session_path,
                    data_path=self.data_path,
                )

            # clear all queues
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            while not self.safety_queue.empty():
                self.safety_queue.get_nowait()

            # prompt engineering
            instruction = await self._build_instruction(
                task=task,
                data_path=self.data_path,
                session_id=self.session_id,
                task_id=task_id,
            )
                
            # Set up Enter-key stop listener for text UI
            if not self.web:
                import sys
                self.chat_ui.console.print(safety_warning, style="dim")
                def _on_enter():
                    sys.stdin.readline()
                    self.safety_queue.put_nowait(STOP_TAG)
                loop.add_reader(sys.stdin.fileno(), _on_enter)

            # submit instruction with retry on API error
            start_time = loop.time()
            while True:
                while not self.safety_queue.empty():
                    self.safety_queue.get_nowait()

                agent_task = asyncio.create_task(self.agent_session())
                await self.input_queue.put({
                    'instruction': instruction,
                    'task': task,
                    'task_id': task_id,
                    'session_id': self.session_id,
                })

                final_answer_task = asyncio.create_task(self.output_queue.get())

                # Watch safety_queue so Enter-key cancellation is detected
                # immediately even while chat() is blocked on an API call.
                async def _safety_watcher():
                    while self.safety_queue.empty():
                        await asyncio.sleep(0.2)
                    return STOP_TAG

                safety_watch_task = asyncio.create_task(_safety_watcher())
                done, pending = await asyncio.wait(
                    [final_answer_task, safety_watch_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

                # Enter pressed — cancel the agent and break
                if safety_watch_task in done and final_answer_task not in done:
                    agent_task.cancel()
                    try:
                        await agent_task
                    except asyncio.CancelledError:
                        pass
                    if self.trace_recorder:
                        self.trace_recorder.record(
                            'agent.task_cancelled',
                            session_id=self.session_id,
                            task_id=task_id,
                            task_preview=task,
                            reason='stopped by user',
                        )
                    self.chat_ui.add_message("system", "Task stopped by user.")
                    break

                if final_answer_task not in done:
                    await self.safety_queue.put(STOP_TAG)
                    while not agent_task.done():
                        await asyncio.sleep(0.1)
                    break

                response = final_answer_task.result()

                # User-initiated stop
                if response == STOP_TAG:
                    if self.trace_recorder:
                        self.trace_recorder.record(
                            'agent.task_cancelled',
                            session_id=self.session_id,
                            task_id=task_id,
                            task_preview=task,
                            reason='stopped by user',
                        )
                    self.chat_ui.add_message("system", "Task stopped by user.")
                    break

                if isinstance(response, tuple) and response[0] is None:
                    _, error_reason = response
                    response = None
                else:
                    error_reason = None

                if response is None:
                    if self.trace_recorder:
                        self.trace_recorder.record(
                            'agent.task_failed',
                            session_id=self.session_id,
                            task_id=task_id,
                            task_preview=task,
                            reason=error_reason or 'agent returned no response',
                        )
                    # API error — ask user whether to retry
                    if not self.web:
                        loop.remove_reader(sys.stdin.fileno())
                    error_detail = f"\nReason: {error_reason}" if error_reason else ""
                    self.chat_ui.add_message("system", f"Unable to get a response from the model.{error_detail}\nWould you like to retry? (yes/no)")
                    if self.web:
                        retry_input = await self.chat_ui.get_user_input_async()
                    else:
                        retry_input = await loop.run_in_executor(
                            None, self.chat_ui.get_user_input)
                    if retry_input.lower().strip() in ('yes', 'y'):
                        if not self.web:
                            loop.add_reader(sys.stdin.fileno(), _on_enter)
                        continue
                    break

                # success
                elapsed_time = loop.time() - start_time
                elapsed_time = f"{elapsed_time:.2f} secs"
                response = remove_tags(response)
                if self.trace_recorder:
                    self.trace_recorder.record(
                        'agent.task_completed',
                        session_id=self.session_id,
                        task_id=task_id,
                        task_preview=task,
                        response_preview=response,
                        response_length=len(response),
                    )
                self.chat_ui.add_message("assistant", response, elapsed=elapsed_time)
                try:
                    with open(self.session_path, "a", encoding="utf-8") as f:
                        session_data = {
                            "task": task,
                            "response": response,
                            "timestamp": loop.time()
                        }
                        f.write(json.dumps(session_data) + "\n")
                except Exception:
                    pass
                break

            # Clean up Enter-key listener for text UI
            if not self.web:
                import sys
                try:
                    loop.remove_reader(sys.stdin.fileno())
                except Exception:
                    pass
            
    async def agent_session(self) -> None:
        """Start the agent session"""
        while True:
            try:
                request = await self.input_queue.get()
                if isinstance(request, dict):
                    instruction = request.get('instruction')
                    task_id = request.get('task_id')
                    session_id = request.get('session_id', self.session_id)
                else:
                    instruction = request
                    task_id = None
                    session_id = self.session_id
                if not self.safety_queue.empty():
                    await self.output_queue.put(STOP_TAG)
                    break
                error_container = []
                kwargs = {'console': self.chat_ui.console,
                          'chat_ui': self.chat_ui,
                          'cursor': AGENT_CURSOR,
                          'memories': None,
                          'verbose': self.verbose,
                          'data_path': self.data_path,
                          'max_tokens': self.model_serving.get('max_tokens', 262144),
                          'session_history': self.load_session_history(),
                          'trace_recorder': self.trace_recorder,
                          'session_id': session_id,
                          'task_id': task_id}
                if self.prompt_intro:
                    kwargs['prompt_intro'] = self.prompt_intro
                last_response = await chat(host=self.model_serving["host"],
                                            host_key=self.model_serving.get("host_key", "EMPTY"),
                                            model=self.model_serving["model"],
                                            instruction=instruction,
                                            tool_registry=self.tool_registry,
                                            safety_queue=self.safety_queue,
                                            think=self.model_serving["think"],
                                            timeout=self.timeout,
                                            error_container=error_container,
                                            **kwargs)
                if last_response is None and self.safety_queue.empty():
                    await self.output_queue.put((None, error_container[0] if error_container else None))
                    return
                if not self.safety_queue.empty():
                    await self.output_queue.put(STOP_TAG)
                    break
                await self.output_queue.put(f"<answer>{last_response}</answer>")
                return
            except asyncio.CancelledError:
                logger.warning("Agent session cancelled.")
                await self.output_queue.put(None)
                return
            except Exception as e:
                logger.error("Error in agent session: %s", e)
                await self.output_queue.put(None)
                return