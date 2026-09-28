import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from anylabeling.services.auto_labeling.__base__ import yolo as yolo_module
from anylabeling.services.auto_labeling import model as model_module


class DummyOnnxBaseModel:
    kpt_shape = "[17, 3]"
    metadata = {}

    def __init__(self, *args, **kwargs):
        pass

    def get_input_shape(self):
        return [1, 3, 640, 640]

    def get_metadata_info(self, field):
        if field == "kpt_shape":
            return self.kpt_shape
        return self.metadata.get(field)


@pytest.fixture
def yolo_without_model(monkeypatch, tmp_path):
    model_file = tmp_path / "model.onnx"
    model_file.touch()
    monkeypatch.setattr(
        yolo_module.YOLO,
        "get_model_abs_path",
        lambda self, config, field: str(model_file),
    )
    monkeypatch.setattr(yolo_module, "OnnxBaseModel", DummyOnnxBaseModel)
    monkeypatch.setattr(model_module, "get_config", lambda: {})
    return yolo_module.YOLO


@pytest.mark.parametrize(
    ("model_type", "expected_task"),
    [
        ("yolo26_det_track", "det"),
        ("yolo26_seg_track", "seg"),
        ("yolo26_obb_track", "obb"),
        ("yolo26_pose_track", "pose"),
    ],
)
def test_yolo26_track_types_use_tracktrack(
    yolo_without_model, model_type, expected_task
):
    classes = {"person": ["nose"]} if expected_task == "pose" else ["person"]
    model = yolo_without_model(
        {
            "type": model_type,
            "name": model_type,
            "display_name": model_type,
            "model_path": "model.onnx",
            "classes": classes,
            "tracker": {
                "tracker_type": "tracktrack",
                "track_high_thresh": 0.6,
                "track_low_thresh": 0.25,
                "new_track_thresh": 0.7,
                "track_buffer": 30,
                "match_thresh": 0.7,
                "lost_match_thr": 0.0,
                "iou_weight": 0.5,
                "conf_weight": 0.1,
                "angle_weight": 0.05,
                "penalty_p": 0.2,
                "reduce_step": 0.05,
                "tai_thr": 0.55,
                "min_track_len": 3,
                "gmc_method": "none",
            },
        },
        on_message=lambda *args, **kwargs: None,
    )

    assert model.task == expected_task
    assert type(model.tracker).__name__ == "TRACKTRACK"


def test_yolo26_raw_segmentation_uses_local_nms_without_shape_error(
    yolo_without_model, monkeypatch
):
    monkeypatch.setattr(
        DummyOnnxBaseModel,
        "metadata",
        {"end2end": "False", "args": "{'nms': None}"},
    )
    model = yolo_without_model(
        {
            "type": "yolo26_seg",
            "name": "building",
            "display_name": "Building",
            "model_path": "model.onnx",
            "classes": ["building"],
        },
        on_message=lambda *args, **kwargs: None,
    )
    model.img_height = model.img_width = 640
    prediction = np.zeros((1, 37, 8400), dtype=np.float32)
    prediction[0, :4, 0] = [320, 320, 200, 200]
    prediction[0, 4, 0] = 0.9
    prediction[0, 5:, 0] = 0.1
    prototypes = np.zeros((1, 32, 160, 160), dtype=np.float32)

    boxes, classes, scores, masks, _keypoints = model.postprocess(
        [prediction, prototypes]
    )

    assert model.prediction_format == "raw"
    assert boxes.shape == (1, 4)
    assert classes.tolist() == [0.0]
    assert scores.tolist() == pytest.approx([0.9])
    assert masks.shape == (1, 640, 640)
    assert "input_iou" in model.get_required_widgets()
    assert "edit_iou" in model.get_required_widgets()


def test_yolo26_end2end_segmentation_hides_local_iou(
    yolo_without_model, monkeypatch
):
    monkeypatch.setattr(
        DummyOnnxBaseModel,
        "metadata",
        {"end2end": "True", "args": "{'nms': False}"},
    )
    model = yolo_without_model(
        {
            "type": "yolo26_seg",
            "name": "building",
            "display_name": "Building",
            "model_path": "model.onnx",
            "classes": ["building"],
        },
        on_message=lambda *args, **kwargs: None,
    )
    model.img_height = model.img_width = 640
    prediction = np.zeros((1, 300, 38), dtype=np.float32)
    prediction[0, 0, :6] = [100, 100, 300, 300, 0.9, 0]
    prediction[0, 0, 6:] = 0.1
    prototypes = np.zeros((1, 32, 160, 160), dtype=np.float32)

    boxes, _classes, _scores, masks, _keypoints = model.postprocess(
        [prediction, prototypes]
    )

    assert model.prediction_format == "end2end"
    assert boxes.shape == (1, 4)
    assert masks.shape == (1, 640, 640)
    assert "input_iou" not in model.get_required_widgets()
    assert "edit_iou" not in model.get_required_widgets()


def test_yolo26_raw_detection_iou_changes_local_nms_result(
    yolo_without_model, monkeypatch
):
    monkeypatch.setattr(
        DummyOnnxBaseModel,
        "metadata",
        {"end2end": "False", "args": "{'nms': None}"},
    )
    model = yolo_without_model(
        {
            "type": "yolo26",
            "name": "building",
            "display_name": "Building",
            "model_path": "model.onnx",
            "classes": ["building"],
        },
        on_message=lambda *args, **kwargs: None,
    )
    model.img_height = model.img_width = 640
    prediction = np.zeros((1, 5, 2), dtype=np.float32)
    prediction[0, :4, 0] = [320, 320, 200, 200]
    prediction[0, :4, 1] = [320, 320, 200, 200]
    prediction[0, 4, :] = [0.9, 0.8]

    model.iou_thres = 0.45
    low_iou_boxes = model.postprocess([prediction.copy()])[0]
    model.iou_thres = 1.0
    high_iou_boxes = model.postprocess([prediction.copy()])[0]

    assert len(low_iou_boxes) == 1
    assert len(high_iou_boxes) == 2


def test_segmentation_points_follow_tracker_detection_indices(
    yolo_without_model, monkeypatch
):
    model = yolo_without_model(
        {
            "type": "yolo11_seg_track",
            "name": "building",
            "display_name": "Building",
            "model_path": "model.onnx",
            "classes": ["building"],
        },
        on_message=lambda *args, **kwargs: None,
    )
    model.tracker = SimpleNamespace(
        update=lambda *_args: np.array(
            [
                [20, 20, 30, 30, 2, 0.8, 0, 1],
                [0, 0, 10, 10, 1, 0.9, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    model.preprocess = lambda *_args, **_kwargs: None
    model.inference = lambda _blob: None
    model.postprocess = lambda _outputs: (
        np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        np.array([0, 0]),
        np.array([0.9, 0.8]),
        np.zeros((2, 2, 2)),
        None,
    )
    model.create_polygon_shape = lambda point, _score, _class_id, track_id: (
        float(point[0, 0]),
        int(track_id),
    )
    monkeypatch.setattr(
        yolo_module,
        "qt_img_to_rgb_cv_img",
        lambda *_args: np.zeros((32, 32, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        yolo_module,
        "masks2segments",
        lambda *_args: [
            np.array([[1, 1], [2, 1], [2, 2]]),
            np.array([[21, 21], [22, 21], [22, 22]]),
        ],
    )
    monkeypatch.setattr(
        yolo_module,
        "scale_coords",
        lambda _shape, points, *_args, **_kwargs: points,
    )

    result = model.predict_shapes(object())

    assert result.shapes == [(21.0, 2), (1.0, 1)]


@pytest.mark.parametrize(
    "metadata",
    [
        "__import__('builtins').sum([20, 22])",
        "[17]",
        "[0, 3]",
        "[17, 4]",
        "[17, 3.0]",
        "[True, 3]",
    ],
)
def test_pose_rejects_invalid_kpt_shape_metadata(
    yolo_without_model, monkeypatch, metadata
):
    monkeypatch.setattr(DummyOnnxBaseModel, "kpt_shape", metadata)

    with pytest.raises(ValueError) as error:
        yolo_without_model(
            {
                "type": "yolo26_pose",
                "name": "yolo26_pose",
                "display_name": "YOLO26 Pose",
                "model_path": "model.onnx",
                "classes": {"person": ["nose"]},
            },
            on_message=lambda *args, **kwargs: None,
        )

    assert "kpt_shape" in str(error.value)
    assert "model.onnx" in str(error.value)


@pytest.mark.parametrize(
    ("config_file", "model_type"),
    [
        ("yolo26s_det_tracktrack.yaml", "yolo26_det_track"),
        ("yolo26s_seg_tracktrack.yaml", "yolo26_seg_track"),
        ("yolo26s_obb_tracktrack.yaml", "yolo26_obb_track"),
        ("yolo26s_pose_tracktrack.yaml", "yolo26_pose_track"),
    ],
)
def test_yolo26_tracktrack_configs_are_registered(config_file, model_type):
    config_path = os.path.join(
        "anylabeling", "configs", "auto_labeling", config_file
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["type"] == model_type
    assert config["tracker"]["tracker_type"] == "tracktrack"

    with open(
        os.path.join("anylabeling", "configs", "models.yaml"),
        "r",
        encoding="utf-8",
    ) as f:
        models = yaml.safe_load(f)

    assert any(item["config_file"] == f":/{config_file}" for item in models)
