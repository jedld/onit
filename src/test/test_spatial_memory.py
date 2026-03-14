"""Tests for src/lib/spatial_memory.py."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.spatial_memory import SpatialMemory


class TestSpatialMemory:
    def test_tracks_pose_and_sensor_snapshot_detections(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        payload = {
            "pose": {"x": 1.0, "y": 2.0, "yaw_deg": 90.0},
            "detections": [
                {
                    "label": "backpack",
                    "confidence": 0.9,
                    "distance_m": 1.0,
                    "bbox": {"cx": 320, "cy": 100, "w": 50, "h": 60},
                }
            ],
        }

        update = memory.observe("get_sensor_snapshot", {}, json.dumps(payload))

        assert update is not None
        assert len(memory.landmarks) == 1
        landmark = memory.landmarks[0]
        assert landmark["normalized_label"] == "backpack"
        assert landmark["estimated_position"] == {"x": 1.0, "y": 3.0}

    def test_deduplicates_same_object_seen_twice(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))

        detection = {
            "label": "door",
            "distance_m": 2.0,
            "bbox": {"cx": 320, "cy": 100, "w": 80, "h": 120},
            "confidence": 0.8,
        }

        memory.observe("detect_objects_in_image", {}, json.dumps([detection]))
        memory.observe("detect_objects_in_image", {}, json.dumps([detection]))

        assert len(memory.landmarks) == 1
        assert memory.landmarks[0]["observation_count"] == 2

    def test_parses_textual_landmarks_from_vision_agent(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))
        response = {
            "vlm_response": """
1. Door 1 (Metal Door)
Position: Centre-right of the frame
Status: Closed

2. Door 2 (Balcony Door)
Position: Far-right side
Status: Open
"""
        }

        update = memory.observe("ask_vision_agent", {}, json.dumps(response))

        assert update is not None
        labels = sorted(landmark["label"] for landmark in memory.landmarks)
        assert labels == ["Balcony Door", "Metal Door"]
        states = {landmark["label"]: landmark["state"] for landmark in memory.landmarks}
        assert states["Metal Door"] == "closed"
        assert states["Balcony Door"] == "open"

    def test_ignores_negative_prose_and_extracts_scene_objects(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        response = {
            "vlm_response": """
Door Identification:
- NOT FOUND - No doors are visible in this scene.

Scene Description:
- A stainless steel trash can on the left side (~0.5m)
- A black folding stool on the right side (~0.5m)
- Light-colored wooden cabinetry/cupboards behind the stool
"""
        }

        update = memory.observe("ask_vision_agent", {}, json.dumps(response))

        assert update is not None
        labels = sorted(landmark["label"] for landmark in memory.landmarks)
        assert labels == [
            "Light-colored wooden cabinetry/cupboards",
            "black folding stool",
            "stainless steel trash can",
        ]
        assert all("no doors" not in label.lower() for label in labels)

    def test_updates_single_door_from_follow_up_state_text(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))

        first_response = {
            "cosmos_response": "Only one door is visible in the scene, located on the left side of the frame. No indication exists that the door is open or closed."
        }
        second_response = {
            "cosmos_response": "The door is likely closed based on the visible structure and lack of any open gap."
        }

        memory.observe("ask_cosmos_agent", {}, json.dumps(first_response))
        memory.observe("ask_cosmos_agent", {}, json.dumps(second_response))

        assert len(memory.landmarks) == 1
        landmark = memory.landmarks[0]
        assert landmark["label"] == "door"
        assert landmark["relative_position"] == "left side of the frame"
        assert landmark["state"] == "closed"

    def test_trims_copula_phrases_from_entity_labels(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        response = {
            "vlm_response": "The door is positioned on the left wall of the room."
        }

        memory.observe("ask_vision_agent", {}, json.dumps(response))

        assert len(memory.landmarks) == 1
        assert memory.landmarks[0]["label"] == "door"