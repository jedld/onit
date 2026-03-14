"""Tests for src/lib/observability.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.observability import TraceRecorder, _dashboard_html


class _FakeRegistry:
    def __init__(self):
        self.tools = {"search", "navigate"}

    def __len__(self):
        return len(self.tools)


class TestTraceRecorder:
    def test_records_events_and_stats(self, tmp_path):
        recorder = TraceRecorder(base_dir=str(tmp_path), enabled=True, max_events=10)

        recorder.record("llm.request", session_id="sess-1", model="qwen", host="http://localhost:8000/v1")
        recorder.record("llm.response", session_id="sess-1", model="qwen", latency_ms=123.4)
        recorder.record("mcp.request", session_id="sess-1", tool_name="navigate", url="http://127.0.0.1:18210/ros2")
        recorder.record("mcp.response", session_id="sess-1", tool_name="navigate", latency_ms=45.6)

        stats = recorder.stats_snapshot()
        assert stats["llm_requests"] == 1
        assert stats["mcp_requests"] == 1
        assert stats["llm_avg_latency_ms"] == 123.4
        assert stats["mcp_avg_latency_ms"] == 45.6
        assert stats["session_activity"]["sess-1"] == 4
        assert os.path.exists(recorder.events_path)

    def test_build_probe_report(self, tmp_path):
        recorder = TraceRecorder(base_dir=str(tmp_path), enabled=True, max_events=10)
        registry = _FakeRegistry()

        report = recorder.build_probe_report(
            model_host="http://127.0.0.1:1/v1",
            mcp_servers=[{"name": "ROS2BridgeMCPServer", "url": "http://127.0.0.1:2/ros2", "enabled": True}],
            tool_registry=registry,
        ).to_dict()

        assert report["model"]["url"] == "http://127.0.0.1:1/v1"
        assert report["mcp_servers"][0]["name"] == "ROS2BridgeMCPServer"
        assert report["tool_registry"]["tool_count"] == 2
        assert sorted(report["tool_registry"]["tools"]) == ["navigate", "search"]

    def test_dashboard_snapshot_tracks_inflight_and_completed(self, tmp_path):
        recorder = TraceRecorder(base_dir=str(tmp_path), enabled=True, max_events=50)

        recorder.record(
            "agent.task_received",
            session_id="sess-1",
            task_id="task-1",
            task_preview="Move robot to waypoint A",
        )
        recorder.record(
            "mcp.request",
            session_id="sess-1",
            task_id="task-1",
            operation_id="mcp-1",
            tool_name="nav2_go_to_pose",
            url="http://robot.local:18210/ros2",
        )

        snapshot = recorder.dashboard_snapshot()
        assert snapshot["stats"]["inflight_task_count"] == 1
        assert snapshot["stats"]["inflight_operation_count"] == 1
        assert snapshot["inflight_tasks"][0]["task_id"] == "task-1"
        assert snapshot["inflight_operations"][0]["operation_id"] == "mcp-1"

        recorder.record(
            "mcp.response",
            session_id="sess-1",
            task_id="task-1",
            operation_id="mcp-1",
            tool_name="nav2_go_to_pose",
            url="http://robot.local:18210/ros2",
            latency_ms=215.0,
        )
        recorder.record(
            "agent.task_completed",
            session_id="sess-1",
            task_id="task-1",
            task_preview="Move robot to waypoint A",
            response_preview="Navigation started",
        )

        snapshot = recorder.dashboard_snapshot()
        assert snapshot["stats"]["inflight_task_count"] == 0
        assert snapshot["stats"]["completed_task_count"] == 1
        assert snapshot["stats"]["completed_operation_count"] == 1
        assert snapshot["completed_tasks"][0]["status"] == "completed"
        assert snapshot["completed_operations"][0]["status"] == "completed"
        assert snapshot["ros_activity"][0]["name"] == "nav2_go_to_pose"

    def test_dashboard_html_contains_live_endpoints(self):
        html = _dashboard_html()
        assert "/dashboard/data" in html
        assert "OnIt ROS/MCP Activity Dashboard" in html