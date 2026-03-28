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

    def test_persists_landmark_hypothesis_fields_for_text_detection(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 1.0, "y": 2.0, "yaw_deg": 90.0}))

        response = {
            "vlm_response": "FOUND. The front side of the toolbox is centered in the frame at about 0.8m."
        }

        memory.observe("ask_vision_agent", {}, json.dumps(response))

        assert len(memory.landmarks) == 1
        landmark = memory.landmarks[0]
        assert landmark["label"] == "toolbox"
        assert landmark["visible_face"] == "front"
        assert landmark["approximate_range_m"] == 0.8
        assert landmark["last_confirmed_pose"] == {"x": 1.0, "y": 2.0, "yaw_deg": 90.0}
        assert landmark["hypothesis"] == {
            "object_label": "toolbox",
            "estimated_bearing_deg": 0.0,
            "approximate_range_m": 0.8,
            "visible_face": "front",
            "relative_position": "centered in the frame at about 0.8m.",
            "last_confirmed_pose": {"x": 1.0, "y": 2.0, "yaw_deg": 90.0},
            "last_confirmed_source": "ask_vision_agent",
            "confidence": 0.7,
        }

    def test_rejects_junk_landmarks_from_negative_vlm_responses(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))

        response = {
            "vlm_response": (
                "NOT FOUND\n\n"
                "There is no television visible in the image. The scene shows:\n"
                "- A black refrigerator on the far-left\n"
                "- A stainless steel trash bin in the center-left\n"
                "- A glossy tiled floor that reflects the objects\n"
                "- The image shows various household items\n"
            ),
        }

        memory.observe("ask_vision_agent", {}, json.dumps(response))

        labels = [lm["label"] for lm in memory.landmarks]
        # Real objects should be kept
        assert "black refrigerator" in labels
        assert "stainless steel trash bin" in labels
        # VLM prose stubs and generic surfaces should be rejected
        for label in labels:
            assert "floor" not in label.lower(), f"floor landmark not rejected: {label}"
            assert "image shows" not in label.lower(), f"prose stub not rejected: {label}"
        # Confidence should be 0.3 (incidental from negative response), not None
        for lm in memory.landmarks:
            assert lm["confidence"] == 0.3, f"Expected 0.3 confidence, got {lm['confidence']}"

    # ---- Vantage point tracking tests ----

    def test_records_vantage_on_sensor_snapshot(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        payload = {
            "pose": {"x": 1.0, "y": 2.0, "yaw_deg": 90.0},
            "detections": [],
        }
        memory.observe("get_sensor_snapshot", {}, json.dumps(payload))

        assert len(memory.visited_vantages) == 1
        assert memory.visited_vantages[0]["x"] == 1.0
        assert memory.visited_vantages[0]["y"] == 2.0

    def test_deduplicates_nearby_vantage_points(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))
        memory.observe("ask_vision_agent", {}, json.dumps({"vlm_response": "A chair on the left side"}))
        # Same position — should not add a second vantage
        memory.observe("ask_vision_agent", {}, json.dumps({"vlm_response": "A table on the right side"}))

        assert len(memory.visited_vantages) == 1

    def test_records_distinct_vantage_points(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))
        memory.observe("ask_vision_agent", {}, json.dumps({"vlm_response": "A chair on the left side"}))
        # Move far enough away — should add a new vantage
        memory.observe("get_robot_pose", {}, json.dumps({"x": 3.0, "y": 0.0, "yaw_deg": 180.0}))
        memory.observe("ask_vision_agent", {}, json.dumps({"vlm_response": "A door ahead"}))

        assert len(memory.visited_vantages) == 2

    def test_describe_scene_records_vantage_and_extracts_landmarks(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 1.0, "y": 1.0, "yaw_deg": 45.0}))
        response = {"vlm_response": "A red toolbox is centered in the frame at about 1.5m."}
        update = memory.observe("describe_scene", {}, json.dumps(response))

        assert len(memory.visited_vantages) == 1
        assert memory.visited_vantages[0]["x"] == 1.0
        assert update is not None
        labels = [lm["label"] for lm in memory.landmarks]
        assert any("toolbox" in label.lower() for label in labels)

    def test_planner_context_includes_visited_vantages(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("get_robot_pose", {}, json.dumps({"x": 0.0, "y": 0.0, "yaw_deg": 0.0}))
        memory.observe(
            "get_sensor_snapshot",
            {},
            json.dumps({
                "pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
                "detections": [{"label": "chair", "distance_m": 1.0, "bbox": {"cx": 320, "cy": 100, "w": 50, "h": 60}, "confidence": 0.8}],
            }),
        )
        context = memory.planner_context()
        assert context is not None
        assert "Visited vantage points" in context
        assert "(0.0, 0.0" in context
        assert "Do NOT revisit" in context

    def test_planner_context_with_only_vantages_no_landmarks(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        # Observe sensor snapshot with empty detections — vantage recorded but no landmarks
        memory.observe(
            "get_sensor_snapshot",
            {},
            json.dumps({"pose": {"x": 2.0, "y": 3.0, "yaw_deg": 90.0}, "detections": []}),
        )
        assert len(memory.visited_vantages) == 1
        assert len(memory.landmarks) == 0
        # should_surface_to_planner should still be True because of vantages
        assert memory.should_surface_to_planner("get_sensor_snapshot") is True
        context = memory.planner_context()
        assert context is not None

    # ── Scan-loop detection tests ──

    def test_scan_counter_increments_on_perception(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        assert memory._scans_since_move == 0
        memory.observe("describe_scene", {}, '{"vlm_response": "A room with a table."}')
        assert memory._scans_since_move == 1
        memory.observe("describe_scene", {}, '{"vlm_response": "A room with a table."}')
        assert memory._scans_since_move == 2

    def test_scan_counter_resets_on_translational_motion(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("describe_scene", {}, '{"vlm_response": "A room."}')
        memory.observe("describe_scene", {}, '{"vlm_response": "A room."}')
        assert memory._scans_since_move == 2
        memory.observe("move_distance", {}, '{"success": true}')
        assert memory._scans_since_move == 0

    def test_scan_counter_not_reset_by_rotation(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("describe_scene", {}, '{"vlm_response": "A room."}')
        memory.observe("describe_scene", {}, '{"vlm_response": "A room."}')
        assert memory._scans_since_move == 2
        memory.observe("rotate_angle", {}, '{"success": true}')
        assert memory._scans_since_move == 2  # rotation does NOT reset

    def test_soft_warning_at_threshold(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        from lib.spatial_memory import SCAN_LOOP_SOFT_THRESHOLD
        for _ in range(SCAN_LOOP_SOFT_THRESHOLD):
            memory.observe("describe_scene", {}, '{"vlm_response": "Nothing here."}')
        warning = memory._scan_loop_warning()
        assert warning is not None
        assert "WARNING" in warning

    def test_hard_warning_at_threshold(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        from lib.spatial_memory import SCAN_LOOP_HARD_THRESHOLD
        for _ in range(SCAN_LOOP_HARD_THRESHOLD):
            memory.observe("describe_scene", {}, '{"vlm_response": "Nothing here."}')
        warning = memory._scan_loop_warning()
        assert warning is not None
        assert "MANDATORY" in warning

    def test_scan_warning_surfaces_in_observe(self, tmp_path):
        """observe() should return the scan-loop warning even when no landmarks changed."""
        memory = SpatialMemory(str(tmp_path))
        from lib.spatial_memory import SCAN_LOOP_SOFT_THRESHOLD
        for _ in range(SCAN_LOOP_SOFT_THRESHOLD):
            result = memory.observe("describe_scene", {}, '{"vlm_response": "Empty room."}')
        # The last observe should return the warning
        assert result is not None
        assert "WARNING" in result

    def test_scan_warning_surfaces_in_planner_context(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        from lib.spatial_memory import SCAN_LOOP_SOFT_THRESHOLD
        # Set pose so vantage is recorded
        memory.observe("get_robot_pose", {}, json.dumps({"x": 1.0, "y": 1.0, "yaw_deg": 0.0}))
        for _ in range(SCAN_LOOP_SOFT_THRESHOLD):
            memory.observe("describe_scene", {}, '{"vlm_response": "Empty room."}')
        context = memory.planner_context()
        assert context is not None
        assert "WARNING" in context or "scan loop" in context.lower()

    def test_navigate_to_pose_resets_scan_counter(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        for _ in range(5):
            memory.observe("describe_scene", {}, '{"vlm_response": "Nothing."}')
        assert memory._scans_since_move == 5
        memory.observe("navigate_to_pose", {}, '{"success": true}')
        assert memory._scans_since_move == 0
        assert memory._scan_loop_warning() is None

    def test_export_includes_scans_since_move(self, tmp_path):
        memory = SpatialMemory(str(tmp_path))
        memory.observe("describe_scene", {}, '{"vlm_response": "A table."}')
        export = memory.export()
        assert "scans_since_move" in export
        assert export["scans_since_move"] == 1