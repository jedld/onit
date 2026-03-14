"""Lightweight spatial memory for accumulating object landmarks across tool calls."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any


DEFAULT_HFOV_DEG = 62.0
MOTION_TOOL_NAMES = {"move_robot", "move_distance", "rotate_angle", "navigate_to_pose", "go_to_waypoint"}


@dataclass
class _Pose:
    x: float
    y: float
    yaw_deg: float


def _safe_json_loads(payload: str) -> Any | None:
    try:
        return json.loads(payload)
    except Exception:
        return None


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (label or "unknown").strip().lower())
    return normalized.strip("_") or "unknown"


def _extract_number(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _extract_state(text: str) -> str | None:
    lowered = (text or "").lower()
    if re.search(r"\b(closed|shut)\b", lowered):
        return "closed"
    if re.search(r"\b(open|opened)\b", lowered):
        return "open"
    return None


def _looks_negative_observation(text: str) -> bool:
    lowered = (text or "").lower()
    negative_patterns = [
        r"\bno doors?\b",
        r"\bno doorways?\b",
        r"\bnot found\b",
        r"\bcannot be identified\b",
        r"\bno additional doors?\b",
        r"\bnot visible\b",
    ]
    return any(re.search(pattern, lowered) for pattern in negative_patterns)


def _clean_landmark_label(label: str) -> str | None:
    cleaned = re.sub(r"\*+", "", label or "").strip(" :-\t")
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"^(door|object)\s+\d+\s*-\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s+(?:is|are|was|were|appears|appear|seems|seem)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    grouped = re.match(r"^(door|object)\s+\d+\s*\(([^)]+)\)$", cleaned, re.IGNORECASE)
    if grouped:
        cleaned = grouped.group(2).strip()
    generic_group = re.match(r"^(door|object)\s*\(([^)]+)\)$", cleaned, re.IGNORECASE)
    if generic_group:
        parenthetical = generic_group.group(2).strip()
        cleaned = generic_group.group(1).strip() if " or " in parenthetical.lower() else parenthetical
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned.split()) > 10:
        return None
    if cleaned.lower().startswith(("the scene", "the space", "the area", "scene overview", "summary")):
        return None
    if _looks_negative_observation(cleaned):
        return None
    return cleaned


def _extract_scene_object(line: str) -> dict[str, Any] | None:
    if not line or _looks_negative_observation(line):
        return None
    match = re.match(
        r"^(?:a|an|the)\s+(?P<label>.+?)(?:\s+(?:on|in|at|near|behind|beside|by|under)\s+(?P<position>.+?))?(?:\s*\((?P<extra>[^)]*)\))?$",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None
    label = _clean_landmark_label(match.group("label") or "")
    if label is None:
        return None
    extra = match.group("extra") or ""
    position = match.group("position") or None
    return {
        "label": label,
        "relative_position": position,
        "distance_m": _extract_number(extra) if extra else None,
        "state": _extract_state(extra) or _extract_state(line),
    }


def _extract_positive_entity_sentence(line: str) -> dict[str, Any] | None:
    if not line or _looks_negative_observation(line):
        return None

    door_match = re.search(
        r"(?:only one|one|a|an)\s+(?P<label>[a-z0-9 /-]*door(?:way)?)\s+is visible(?: in (?:the|this) (?:scene|view)| in the current camera view)?(?:,?\s*located on the\s+(?P<position>[^.]+))?",
        line,
        re.IGNORECASE,
    )
    if door_match:
        label = _clean_landmark_label(door_match.group("label") or "door")
        if label:
            return {
                "label": label,
                "relative_position": door_match.group("position") or None,
                "distance_m": _extract_number(line),
                "state": _extract_state(line),
            }

    heading_match = re.match(r"^(?P<label>[^:]+door(?:way)?(?: to [^:]+)?)\s*:\s*$", line, re.IGNORECASE)
    if heading_match:
        label = _clean_landmark_label(heading_match.group("label"))
        if label:
            return {"label": label, "relative_position": None, "distance_m": None, "state": None}
    return None


def _keyword_bearing_deg(position_text: str) -> float | None:
    text = (position_text or "").lower()
    mapping = [
        ("far-left", -50.0),
        ("centre-left", -20.0),
        ("center-left", -20.0),
        ("left", -30.0),
        ("centre-right", 20.0),
        ("center-right", 20.0),
        ("far-right", 50.0),
        ("right", 30.0),
        ("centre", 0.0),
        ("center", 0.0),
        ("middle", 0.0),
    ]
    for key, value in mapping:
        if key in text:
            return value
    return None


def _position_from_pose(pose: _Pose | None, distance_m: float | None, bearing_deg: float | None) -> dict[str, float] | None:
    if pose is None or distance_m is None:
        return None
    heading_rad = math.radians(pose.yaw_deg + (bearing_deg or 0.0))
    return {
        "x": round(pose.x + distance_m * math.cos(heading_rad), 3),
        "y": round(pose.y + distance_m * math.sin(heading_rad), 3),
    }


class SpatialMemory:
    """Task-scoped spatial memory that clusters object observations into landmarks."""

    def __init__(self,
                 data_path: str,
                 trace_recorder=None,
                 session_id: str | None = None,
                 task_id: str | None = None) -> None:
        self.data_path = data_path
        self.trace_recorder = trace_recorder
        self.session_id = session_id
        self.task_id = task_id
        self.state_path = os.path.join(data_path, "spatial_memory.json") if data_path else ""
        self.landmarks: list[dict[str, Any]] = []
        self.last_pose: _Pose | None = None
        self.camera_hfov_deg = DEFAULT_HFOV_DEG
        self.image_width_px = 640.0
        if self.state_path:
            os.makedirs(data_path, exist_ok=True)

    def observe(self, function_name: str, function_arguments: dict[str, Any], tool_response: str) -> str | None:
        """Update memory from a tool response and return a short summary when new landmarks were added/updated."""
        parser = getattr(self, f"_observe_{function_name}", None)
        if parser is None:
            return None
        before = len(self.landmarks)
        changed_labels = parser(function_arguments, tool_response) or []
        if not changed_labels:
            return None
        self._persist()
        if self.trace_recorder:
            self.trace_recorder.record(
                "agent.spatial_memory_updated",
                session_id=self.session_id,
                task_id=self.task_id,
                tool_name=function_name,
                landmarks=len(self.landmarks),
                changed=changed_labels,
            )
        action = "added" if len(self.landmarks) > before else "updated"
        return (
            f"[Spatial memory {action}: {', '.join(changed_labels)}. "
            f"Tracked landmarks: {self.summary(max_items=6)}]"
        )

    def summary(self, max_items: int = 8) -> str:
        if not self.landmarks:
            return "none"
        items = []
        for landmark in self.landmarks[:max_items]:
            position = landmark.get("estimated_position")
            status = landmark.get("state")
            where = f" @ ({position['x']}, {position['y']})" if position else ""
            suffix = f" [{status}]" if status else ""
            items.append(f"{landmark['label']}{suffix}{where}")
        return "; ".join(items)

    def export(self) -> dict[str, Any]:
        return {
            "camera_hfov_deg": self.camera_hfov_deg,
            "image_width_px": self.image_width_px,
            "last_pose": None if self.last_pose is None else self.last_pose.__dict__,
            "landmarks": self.landmarks,
        }

    def _persist(self) -> None:
        if not self.state_path:
            return
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(self.export(), handle, indent=2)

    def _observe_get_robot_pose(self, _arguments: dict[str, Any], tool_response: str) -> list[str]:
        payload = _safe_json_loads(tool_response)
        if not isinstance(payload, dict):
            return []
        yaw_deg = payload.get("yaw_deg")
        if yaw_deg is None and payload.get("yaw_rad") is not None:
            yaw_deg = math.degrees(float(payload["yaw_rad"]))
        if payload.get("x") is None or payload.get("y") is None or yaw_deg is None:
            return []
        self.last_pose = _Pose(float(payload["x"]), float(payload["y"]), float(yaw_deg))
        return []

    def _observe_get_sensor_snapshot(self, _arguments: dict[str, Any], tool_response: str) -> list[str]:
        payload = _safe_json_loads(tool_response)
        if not isinstance(payload, dict):
            return []
        pose = payload.get("pose") or {}
        self._observe_get_robot_pose({}, json.dumps(pose))
        changed = []
        detections = payload.get("detections") or []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            changed_label = self._upsert_landmark(
                label=detection.get("label", "object"),
                source="get_sensor_snapshot",
                distance_m=detection.get("distance_m"),
                bearing_deg=self._bearing_from_bbox(detection.get("bbox")),
                bbox=detection.get("bbox"),
                raw=detection,
                confidence=detection.get("confidence"),
            )
            if changed_label:
                changed.append(changed_label)
        return changed

    def _observe_detect_objects_in_image(self, _arguments: dict[str, Any], tool_response: str) -> list[str]:
        payload = _safe_json_loads(tool_response)
        if isinstance(payload, dict):
            detections = payload.get("detections") or payload.get("objects") or []
        elif isinstance(payload, list):
            detections = payload
        else:
            detections = []
        changed = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            changed_label = self._upsert_landmark(
                label=detection.get("label", "object"),
                source="detect_objects_in_image",
                distance_m=detection.get("distance_m"),
                bearing_deg=self._bearing_from_bbox(detection.get("bbox")),
                bbox=detection.get("bbox"),
                raw=detection,
                confidence=detection.get("confidence"),
            )
            if changed_label:
                changed.append(changed_label)
        return changed

    def _observe_get_camera_info(self, _arguments: dict[str, Any], tool_response: str) -> list[str]:
        payload = _safe_json_loads(tool_response)
        if not isinstance(payload, dict):
            return []
        width = payload.get("width") or payload.get("image_width")
        if width:
            self.image_width_px = float(width)
        hfov = payload.get("horizontal_fov_deg") or payload.get("hfov_deg")
        if hfov:
            self.camera_hfov_deg = float(hfov)
        return []

    def _observe_ask_vision_agent(self, _arguments: dict[str, Any], tool_response: str) -> list[str]:
        return self._observe_textual_scene("ask_vision_agent", tool_response)

    def _observe_ask_cosmos_agent(self, _arguments: dict[str, Any], tool_response: str) -> list[str]:
        return self._observe_textual_scene("ask_cosmos_agent", tool_response)

    def _observe_move_robot(self, _arguments: dict[str, Any], _tool_response: str) -> list[str]:
        self.last_pose = None
        return []

    def _observe_move_distance(self, _arguments: dict[str, Any], _tool_response: str) -> list[str]:
        self.last_pose = None
        return []

    def _observe_rotate_angle(self, _arguments: dict[str, Any], _tool_response: str) -> list[str]:
        self.last_pose = None
        return []

    def _observe_navigate_to_pose(self, _arguments: dict[str, Any], _tool_response: str) -> list[str]:
        self.last_pose = None
        return []

    def _observe_go_to_waypoint(self, _arguments: dict[str, Any], _tool_response: str) -> list[str]:
        self.last_pose = None
        return []

    def _observe_textual_scene(self, source: str, tool_response: str) -> list[str]:
        payload = _safe_json_loads(tool_response)
        if isinstance(payload, dict):
            if payload.get("error"):
                return []
            text = payload.get("vlm_response") or payload.get("cosmos_response") or payload.get("result") or tool_response
        else:
            text = tool_response
        changed = self._extract_text_landmarks(text, source=source)
        if changed:
            return changed
        return self._update_single_door_state(text, source=source)

    def _extract_text_landmarks(self, text: str, source: str) -> list[str]:
        if not text:
            return []
        lines = [re.sub(r"\*+", "", line).strip(" -\t") for line in text.splitlines() if line.strip()]
        candidates: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        def flush() -> None:
            nonlocal current
            if not current:
                return
            label = _clean_landmark_label(current.get("label", ""))
            if label:
                candidates.append({
                    "label": label,
                    "relative_position": current.get("relative_position"),
                    "distance_m": current.get("distance_m"),
                    "state": current.get("state"),
                })
            current = None

        for line in lines:
            if not line or line.startswith("["):
                continue

            if current is None:
                direct_candidate = _extract_positive_entity_sentence(line) or _extract_scene_object(line)
                if direct_candidate:
                    candidates.append(direct_candidate)
                    continue

            structured = re.match(r"^(?:\d+\.\s*)?Door\s+\d+\s*\(([^)]+)\)\s*$", line, re.IGNORECASE)
            if structured:
                flush()
                current = {"label": structured.group(1), "relative_position": None, "distance_m": None, "state": None}
                continue
            if line.lower().startswith("object:"):
                flush()
                current = {"label": line.split(":", 1)[1].strip(), "relative_position": None, "distance_m": None, "state": None}
                continue
            if current and line.lower().startswith("position:"):
                current["relative_position"] = line.split(":", 1)[1].strip()
                continue
            if current and (line.lower().startswith("status:") or line.lower().startswith("state:")):
                current["state"] = _extract_state(line.split(":", 1)[1].strip().strip("*"))
                continue
            if current and "distance" in line.lower() and current.get("distance_m") is None:
                current["distance_m"] = _extract_number(line)
                continue
            if current and _looks_negative_observation(line):
                continue

        flush()

        if len(candidates) == 1 and not candidates[0].get("state"):
            candidates[0]["state"] = _extract_state(text)

        changed = []
        for candidate in candidates:
            changed_label = self._upsert_landmark(
                label=candidate["label"],
                source=source,
                distance_m=candidate.get("distance_m"),
                bearing_deg=_keyword_bearing_deg(candidate.get("relative_position")),
                relative_position=candidate.get("relative_position"),
                state=candidate.get("state"),
                raw={"text": text[:400]},
            )
            if changed_label:
                changed.append(changed_label)
        return changed

    def _update_single_door_state(self, text: str, source: str) -> list[str]:
        state = _extract_state(text)
        if state is None or not re.search(r"\bthe door\b", text.lower()):
            return []
        door_landmarks = [landmark for landmark in self.landmarks if "door" in (landmark.get("normalized_label") or "")]
        if len(door_landmarks) != 1:
            return []
        landmark = door_landmarks[0]
        landmark["state"] = state
        landmark["observation_count"] += 1
        if source not in landmark["sources"]:
            landmark["sources"].append(source)
        landmark["observations"].append({
            "source": source,
            "distance_m": None,
            "bearing_deg": None,
            "bbox": None,
            "relative_position": landmark.get("relative_position"),
            "state": state,
            "pose": None if self.last_pose is None else self.last_pose.__dict__,
            "raw": {"text": text[:400]},
        })
        landmark["observations"] = landmark["observations"][-8:]
        return [landmark["label"]]

    def _bearing_from_bbox(self, bbox: dict[str, Any] | None) -> float | None:
        if not isinstance(bbox, dict):
            return None
        cx = bbox.get("cx")
        if cx is None:
            return None
        normalized = (float(cx) / self.image_width_px) - 0.5
        return round(normalized * self.camera_hfov_deg, 2)

    def _upsert_landmark(self,
                         label: str,
                         source: str,
                         distance_m: float | None = None,
                         bearing_deg: float | None = None,
                         bbox: dict[str, Any] | None = None,
                         relative_position: str | None = None,
                         state: str | None = None,
                         raw: Any | None = None,
                         confidence: float | None = None) -> str | None:
        normalized = _normalize_label(label)
        estimated_position = _position_from_pose(self.last_pose, float(distance_m) if distance_m is not None else None, bearing_deg)
        landmark = self._match_landmark(normalized, estimated_position)
        if landmark is None:
            landmark = {
                "id": f"landmark_{uuid.uuid4().hex[:12]}",
                "label": label.strip(),
                "normalized_label": normalized,
                "estimated_position": estimated_position,
                "relative_position": relative_position,
                "state": state.lower() if isinstance(state, str) else None,
                "confidence": confidence,
                "observation_count": 0,
                "sources": [],
                "observations": [],
            }
            self.landmarks.append(landmark)
        elif estimated_position and landmark.get("estimated_position"):
            old = landmark["estimated_position"]
            landmark["estimated_position"] = {
                "x": round((old["x"] + estimated_position["x"]) / 2.0, 3),
                "y": round((old["y"] + estimated_position["y"]) / 2.0, 3),
            }
        elif estimated_position and not landmark.get("estimated_position"):
            landmark["estimated_position"] = estimated_position

        landmark["label"] = landmark.get("label") or label.strip()
        landmark["relative_position"] = relative_position or landmark.get("relative_position")
        if state:
            landmark["state"] = state.lower()
        if confidence is not None:
            landmark["confidence"] = max(float(confidence), float(landmark.get("confidence") or 0.0))
        landmark["observation_count"] += 1
        if source not in landmark["sources"]:
            landmark["sources"].append(source)
        landmark["observations"].append({
            "source": source,
            "distance_m": distance_m,
            "bearing_deg": bearing_deg,
            "bbox": bbox,
            "relative_position": relative_position,
            "state": state,
            "pose": None if self.last_pose is None else self.last_pose.__dict__,
            "raw": raw,
        })
        landmark["observations"] = landmark["observations"][-8:]
        return landmark["label"]

    def _match_landmark(self, normalized_label: str, estimated_position: dict[str, float] | None) -> dict[str, Any] | None:
        candidates = [landmark for landmark in self.landmarks if landmark.get("normalized_label") == normalized_label]
        if not candidates:
            return None
        if estimated_position is None:
            return candidates[0]
        best = None
        best_distance = None
        for candidate in candidates:
            pos = candidate.get("estimated_position")
            if pos is None:
                return candidate
            distance = math.hypot(pos["x"] - estimated_position["x"], pos["y"] - estimated_position["y"])
            if best_distance is None or distance < best_distance:
                best = candidate
                best_distance = distance
        if best_distance is not None and best_distance <= 1.0:
            return best
        return None