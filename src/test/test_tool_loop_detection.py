"""Tests for _detect_tool_name_loop logic.

chat.py cannot be imported directly in tests due to relative import issues,
so we replicate the function here for isolated testing.
"""

import json


def _detect_tool_name_loop(history: list, min_repeats: int = 4) -> str | None:
    """Detect if recent tool calls form a short repeating name pattern.

    Mirror of the function in model/serving/chat.py.
    """
    names = [name for name, _ in history]
    if len(names) < min_repeats * 2:
        return None
    pair = (names[-2], names[-1])
    count = 0
    idx = len(names) - 2
    while idx >= 0 and idx + 1 < len(names):
        if (names[idx], names[idx + 1]) == pair:
            count += 1
            idx -= 2
        else:
            break
    if count >= min_repeats:
        return f"{pair[0]} → {pair[1]} (repeated {count} times)"
    return None


class TestDetectToolNameLoop:
    def test_no_loop_short_history(self):
        history = [("describe_scene", "{}"), ("rotate_angle", '{"angle": 45}')]
        assert _detect_tool_name_loop(history, min_repeats=4) is None

    def test_no_loop_varied_calls(self):
        history = [
            ("describe_scene", "{}"),
            ("rotate_angle", '{"angle": 45}'),
            ("get_robot_pose", "{}"),
            ("navigate_to_pose", '{"x": 1}'),
            ("describe_scene", "{}"),
            ("get_laser_scan", "{}"),
            ("describe_scene", "{}"),
            ("navigate_to_pose", '{"x": 2}'),
        ]
        assert _detect_tool_name_loop(history, min_repeats=4) is None

    def test_detects_alternating_pair_loop(self):
        # describe_scene → rotate_angle repeating 5 times = 10 calls
        history = []
        for _ in range(5):
            history.append(("describe_scene", '{"question": "find TV"}'))
            history.append(("rotate_angle", '{"angle": -45}'))
        result = _detect_tool_name_loop(history, min_repeats=4)
        assert result is not None
        assert "describe_scene" in result
        assert "rotate_angle" in result

    def test_detects_at_exact_threshold(self):
        history = []
        for _ in range(4):
            history.append(("describe_scene", "{}"))
            history.append(("rotate_angle", "{}"))
        result = _detect_tool_name_loop(history, min_repeats=4)
        assert result is not None

    def test_no_detection_below_threshold(self):
        history = []
        for _ in range(3):
            history.append(("describe_scene", "{}"))
            history.append(("rotate_angle", "{}"))
        result = _detect_tool_name_loop(history, min_repeats=4)
        assert result is None

    def test_detects_with_preceding_different_calls(self):
        # Some initial varied calls, then the loop starts
        history = [
            ("get_robot_pose", "{}"),
            ("get_laser_scan", "{}"),
            ("navigate_to_pose", "{}"),
        ]
        for _ in range(5):
            history.append(("describe_scene", "{}"))
            history.append(("rotate_angle", "{}"))
        result = _detect_tool_name_loop(history, min_repeats=4)
        assert result is not None

    def test_different_args_still_detected(self):
        """The function checks names only, not arguments."""
        history = []
        for i in range(5):
            history.append(("describe_scene", json.dumps({"question": f"q{i}"})))
            history.append(("rotate_angle", json.dumps({"angle": -45 + i * 10})))
        result = _detect_tool_name_loop(history, min_repeats=4)
        assert result is not None
