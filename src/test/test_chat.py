"""Tests for src/model/serving/chat.py — _resolve_api_key, _parse_tool_call_from_content, chat."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.observability import TraceRecorder
from model.serving.chat import (
    _derive_retry_max_tokens_from_context_error,
    _resolve_api_key,
    _parse_tool_call_from_content,
    _execute_tool_call,
    chat,
)


# ── _resolve_api_key ────────────────────────────────────────────────────────

class TestResolveApiKey:
    def test_vllm_returns_host_key(self):
        assert _resolve_api_key("http://localhost:8000/v1", "EMPTY") == "EMPTY"

    def test_openrouter_with_host_key(self):
        assert _resolve_api_key("https://openrouter.ai/api/v1", "sk-or-abc") == "sk-or-abc"

    def test_openrouter_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-key")
        assert _resolve_api_key("https://openrouter.ai/api/v1") == "sk-env-key"

    def test_openrouter_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenRouter requires"):
            _resolve_api_key("https://openrouter.ai/api/v1")


# ── _parse_tool_call_from_content ───────────────────────────────────────────

class TestParseToolCallFromContent:
    def _registry(self, names):
        reg = MagicMock()
        reg.tools = set(names)
        return reg

    def test_valid_tool_call(self):
        content = '{"name": "search", "arguments": {"query": "test"}}'
        result = _parse_tool_call_from_content(content, self._registry(["search"]))
        assert result["name"] == "search"
        assert result["arguments"]["query"] == "test"

    def test_with_think_tags(self):
        content = '<think>thinking...</think>{"name": "search", "arguments": {"q": "x"}}'
        result = _parse_tool_call_from_content(content, self._registry(["search"]))
        assert result is not None
        assert result["name"] == "search"

    def test_unknown_tool_returns_none(self):
        content = '{"name": "unknown_tool", "arguments": {}}'
        result = _parse_tool_call_from_content(content, self._registry(["search"]))
        assert result is None

    def test_no_json_returns_none(self):
        result = _parse_tool_call_from_content("just plain text", self._registry(["search"]))
        assert result is None

    def test_malformed_json_returns_none(self):
        result = _parse_tool_call_from_content('{"name": broken}', self._registry(["search"]))
        assert result is None

    def test_empty_content_returns_none(self):
        assert _parse_tool_call_from_content("", self._registry(["x"])) is None

    def test_none_content_returns_none(self):
        assert _parse_tool_call_from_content(None, self._registry(["x"])) is None

    def test_none_registry_returns_none(self):
        assert _parse_tool_call_from_content('{"name":"x","arguments":{}}', None) is None


# ── chat (async) ────────────────────────────────────────────────────────────

def _mock_completion(content="Hello!", tool_calls=None):
    """Build a mock chat completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 0
    return completion


def _mock_tool_call(name="search", arguments='{"query": "test"}', call_id="call_123"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.id = call_id
    return tc


class TestChat:
    @pytest.mark.asyncio
    async def test_simple_response(self):
        """No tools, model returns plain text."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("The answer is 42.")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="What is 6*7?",
                safety_queue=asyncio.Queue(),
            )

        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_strips_think_tags(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("<think>pondering</think>Result here")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=asyncio.Queue(),
            )

        assert result == "Result here"

    @pytest.mark.asyncio
    async def test_safety_queue_aborts(self):
        sq = asyncio.Queue()
        sq.put_nowait("stop")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("should not reach")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=sq,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_api_timeout_returns_none(self):
        from openai import APITimeoutError

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=asyncio.Queue(),
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_openai_error_returns_none(self):
        from openai import OpenAIError

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=OpenAIError("fail")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="test",
                safety_queue=asyncio.Queue(),
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_tool_calling_loop(self):
        """Model requests a tool, gets result, then gives final answer."""
        tc = _mock_tool_call("search", '{"query": "weather"}')

        # First call: model returns tool_calls
        first_completion = _mock_completion(content=None, tool_calls=[tc])
        # Second call: model returns final answer
        second_completion = _mock_completion("It's sunny!")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[first_completion, second_completion]
        )

        # Mock tool registry
        mock_handler = AsyncMock(return_value="Weather: sunny, 25C")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [{"type": "function", "function": {"name": "search"}}]
        mock_registry.tools = {"search"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            result = await chat(
                host="http://localhost:8000/v1",
                instruction="What is the weather?",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
            )

        assert result == "It's sunny!"
        mock_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trace_recorder_captures_llm_and_tool_events(self, tmp_path):
        tc = _mock_tool_call("search", '{"query": "weather"}')
        first_completion = _mock_completion(content=None, tool_calls=[tc])
        second_completion = _mock_completion("It's sunny!")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[first_completion, second_completion]
        )

        mock_handler = AsyncMock(return_value="Weather: sunny, 25C")
        mock_registry = MagicMock()
        mock_registry.get_tool_items.return_value = [{"type": "function", "function": {"name": "search"}}]
        mock_registry.tools = {"search"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        recorder = TraceRecorder(base_dir=str(tmp_path), enabled=True)

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client):
            await chat(
                host="http://localhost:8000/v1",
                model="test",
                instruction="What is the weather?",
                tool_registry=mock_registry,
                safety_queue=asyncio.Queue(),
                trace_recorder=recorder,
                session_id="chat-test",
            )

        event_types = [event["event_type"] for event in recorder.recent_events(limit=10)]
        assert "llm.request" in event_types
        assert "llm.response" in event_types
        assert "agent.tool_decision" in event_types

    @pytest.mark.asyncio
    async def test_execute_tool_call_appends_spatial_memory_update(self, tmp_path):
        mock_handler = AsyncMock(return_value=json.dumps({
            "pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
            "detections": [{
                "label": "chair",
                "distance_m": 1.2,
                "bbox": {"cx": 320, "cy": 120, "w": 80, "h": 120},
                "confidence": 0.9,
            }],
        }))
        mock_registry = MagicMock()
        mock_registry.tools = {"get_sensor_snapshot"}
        mock_registry.__getitem__ = MagicMock(return_value=mock_handler)

        from lib.spatial_memory import SpatialMemory
        spatial_memory = SpatialMemory(str(tmp_path))

        tool_message = await _execute_tool_call(
            function_name="get_sensor_snapshot",
            function_arguments={},
            tool_call_id="call_1",
            tool_registry=mock_registry,
            timeout=30,
            data_path=str(tmp_path),
            chat_ui=None,
            verbose=False,
            trace_recorder=None,
            session_id="sess-1",
            task_id="task-1",
            spatial_memory=spatial_memory,
        )

        assert "Spatial memory" in tool_message["content"]
        assert len(spatial_memory.landmarks) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_call_refreshes_pose_after_motion(self, tmp_path):
        move_handler = AsyncMock(return_value=json.dumps({"status": "succeeded", "distance_actual": 0.5}))
        pose_handler = AsyncMock(return_value=json.dumps({"x": 1.25, "y": -0.5, "yaw_deg": 90.0}))

        mock_registry = MagicMock()
        mock_registry.tools = {"move_distance", "get_robot_pose"}
        mock_registry.__getitem__ = MagicMock(side_effect=lambda name: {
            "move_distance": move_handler,
            "get_robot_pose": pose_handler,
        }[name])

        from lib.spatial_memory import SpatialMemory
        spatial_memory = SpatialMemory(str(tmp_path))

        await _execute_tool_call(
            function_name="move_distance",
            function_arguments={"distance": 0.5},
            tool_call_id="call_motion",
            tool_registry=mock_registry,
            timeout=30,
            data_path=str(tmp_path),
            chat_ui=None,
            verbose=False,
            trace_recorder=None,
            session_id="sess-1",
            task_id="task-1",
            spatial_memory=spatial_memory,
        )

        pose_handler.assert_awaited_once()
        assert spatial_memory.last_pose is not None
        assert spatial_memory.last_pose.x == 1.25
        assert spatial_memory.last_pose.y == -0.5
        assert spatial_memory.last_pose.yaw_deg == 90.0

    @pytest.mark.asyncio
    async def test_execute_tool_call_appends_landmark_hypothesis_context(self, tmp_path):
        pose_handler = AsyncMock(return_value=json.dumps({"x": 1.0, "y": 2.0, "yaw_deg": 45.0}))

        mock_registry = MagicMock()
        mock_registry.tools = {"get_robot_pose"}
        mock_registry.__getitem__ = MagicMock(return_value=pose_handler)

        from lib.spatial_memory import SpatialMemory
        spatial_memory = SpatialMemory(str(tmp_path))
        spatial_memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))
        spatial_memory.observe(
            "ask_vision_agent",
            {},
            json.dumps({"vlm_response": "FOUND. The front side of the toolbox is centered in the frame at about 0.8m."}),
        )

        tool_message = await _execute_tool_call(
            function_name="get_robot_pose",
            function_arguments={},
            tool_call_id="call_pose",
            tool_registry=mock_registry,
            timeout=30,
            data_path=str(tmp_path),
            chat_ui=None,
            verbose=False,
            trace_recorder=None,
            session_id="sess-1",
            task_id="task-1",
            spatial_memory=spatial_memory,
        )

        assert "Persistent landmark hypotheses" in tool_message["content"]
        assert "label=toolbox" in tool_message["content"]

    @pytest.mark.asyncio
    async def test_session_history_injected(self):
        """Session history entries are added as user/assistant message pairs."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        history = [{"task": "prior question", "response": "prior answer"}]

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            await chat(
                host="http://localhost:8000/v1",
                instruction="follow up",
                safety_queue=asyncio.Queue(),
                session_history=history,
            )

        # Verify the messages list included session history before the current instruction
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # Should have: system, history user, history assistant, current user instruction
        contents = [m.get("content", "") for m in messages if isinstance(m, dict)]
        assert "prior question" in contents
        assert "prior answer" in contents
        # Current instruction must be the last user message
        assert messages[-1]["content"] == "follow up"
        assert messages[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_custom_prompt_intro(self):
        """Custom prompt_intro overrides the default system message."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
             patch("model.serving.chat._resolve_model_id", new_callable=AsyncMock, return_value="test-model"):
            await chat(
                host="http://localhost:8000/v1",
                instruction="hello",
                safety_queue=asyncio.Queue(),
                prompt_intro="I am a custom bot.",
            )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "custom bot" in system_content
        assert "OnIt" not in system_content

    @pytest.mark.asyncio
    async def test_does_not_inject_empty_tool_message(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok")
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client):
            await chat(
                host="http://localhost:8000/v1",
                model="test",
                instruction="hello",
                safety_queue=asyncio.Queue(),
            )

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert all(message["role"] != "tool" for message in messages)

    @pytest.mark.asyncio
    async def test_retries_with_reduced_max_tokens_on_vllm_context_budget_error(self):
        from openai import OpenAIError

        error_message = (
            "You passed 4353 input tokens and requested 12032 output tokens. However, the model's "
            "context length is only 16384 tokens, resulting in a maximum input length of 4352 tokens. "
            "Please reduce the length of the input prompt. (parameter=input_tokens, value=4353)"
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[OpenAIError(error_message), _mock_completion("Recovered")]
        )

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client):
            result = await chat(
                host="http://localhost:8000/v1",
                model="test",
                instruction="hello",
                safety_queue=asyncio.Queue(),
                max_tokens=12032,
            )

        assert result == "Recovered"
        first_call = mock_client.chat.completions.create.call_args_list[0].kwargs
        second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert first_call["max_tokens"] == 12032
        assert second_call["max_tokens"] == 11999

    @pytest.mark.asyncio
    async def test_trims_oldest_history_turn_on_context_budget_retry(self):
        from openai import OpenAIError

        error_message = (
            "You passed 15523 input tokens and requested 862 output tokens. However, the model's "
            "context length is only 16384 tokens, resulting in a maximum input length of 15522 tokens. "
            "Please reduce the length of the input prompt. (parameter=input_tokens, value=15523)"
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[OpenAIError(error_message), _mock_completion("Recovered")]
        )

        history = [
            {"task": "old question", "response": "old answer"},
            {"task": "recent question", "response": "recent answer"},
        ]

        with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client):
            result = await chat(
                host="http://localhost:8000/v1",
                model="test",
                instruction="hello",
                safety_queue=asyncio.Queue(),
                session_history=history,
                max_tokens=862,
            )

        assert result == "Recovered"
        second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
        second_messages = second_call["messages"]
        second_contents = [m.get("content", "") for m in second_messages if isinstance(m, dict)]
        assert "old question" not in second_contents
        assert "old answer" not in second_contents
        assert "recent question" in second_contents
        assert "recent answer" in second_contents
        assert second_call["max_tokens"] == 829


class TestContextBudgetParsing:
    def test_parses_vllm_context_budget_error(self):
        error_message = (
            "You passed 4353 input tokens and requested 12032 output tokens. However, the model's "
            "context length is only 16384 tokens, resulting in a maximum input length of 4352 tokens."
        )

        result = _derive_retry_max_tokens_from_context_error(error_message, 12032)

        assert result == (16384, 4353, 12032, 11999)

    def test_parses_zero_output_budget_context_error(self):
        error_message = (
            "You passed 32768 input tokens and requested 1 output tokens. However, the model's "
            "context length is only 32768 tokens, resulting in a maximum input length of 32767 tokens."
        )

        result = _derive_retry_max_tokens_from_context_error(error_message, 1)

        assert result == (32768, 32768, 1, 1)
