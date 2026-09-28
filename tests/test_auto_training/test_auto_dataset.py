import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anylabeling.services.auto_training.dataset_inspection import (
    inspect_annotations,
)
from anylabeling.services.auto_training.ultralytics import general
from anylabeling.services.auto_training.ultralytics._io import load_yaml_config
from anylabeling.services.auto_training.ultralytics.validators import (
    validate_basic_config,
)
from anylabeling.views.training.ultralytics_dialog import UltralyticsDialog


def _annotated_image(folder, name, labels, checked=True):
    image = folder / f"{name}.jpg"
    image.write_bytes(b"image")
    annotation = {
        "imageWidth": 32,
        "imageHeight": 32,
        "checked": checked,
        "shapes": [
            {
                "shape_type": "polygon",
                "label": label,
                "points": [[1, 1], [25, 1], [25, 25]],
            }
            for label in labels
        ],
    }
    image.with_suffix(".json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    return str(image)


def test_annotation_scan_respects_task_and_review_state(tmp_path):
    first = _annotated_image(tmp_path, "one", ["building", "car"])
    second = _annotated_image(tmp_path, "two", ["building"], checked=False)
    third = _annotated_image(tmp_path, "three", ["car"])

    result = inspect_annotations(
        [first, second, third], ["polygon"], only_checked=True
    )

    assert result.annotation_counts == {"building": 1, "car": 2}
    assert result.images_with_any(["building"]) == 1
    assert result.unchecked_annotations == 1


def test_auto_dataset_generates_yaml_and_only_selected_labels(
    tmp_path, monkeypatch
):
    images = [
        _annotated_image(tmp_path, "one", ["building", "car"]),
        _annotated_image(tmp_path, "two", ["building"]),
        _annotated_image(tmp_path, "three", ["car"]),
    ]
    dataset_root = tmp_path / "datasets"
    monkeypatch.setattr(general, "get_dataset_path", lambda: str(dataset_root))

    model = tmp_path / "base.pt"
    model.write_bytes(b"checkpoint")
    config = {
        "basic": {
            "project": str(tmp_path / "runs"),
            "name": "exp",
            "model": str(model),
            "data": "",
        }
    }
    assert validate_basic_config(config, allow_generated_data=True) == (
        True,
        "",
    )

    output = general.create_yolo_dataset(
        images, "Segment", 0.5, "", class_names=["building"]
    )
    data = load_yaml_config(str(output) + "/data.yaml")
    assert data["names"] == {0: "building"}
    assert data["nc"] == 1
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"

    labels = list(Path(output).glob("labels/*/*.txt"))
    assert len(labels) == 2
    assert all(
        label.read_text(encoding="utf-8").startswith("0 ") for label in labels
    )
    assert not list(Path(output).glob("images/*/three.jpg"))


def test_auto_training_args_use_generated_yaml_instead_of_saved_yaml(
    tmp_path, monkeypatch
):
    images = [
        _annotated_image(tmp_path, "one", ["building"]),
        _annotated_image(tmp_path, "two", ["building"]),
    ]
    monkeypatch.setattr(
        general, "get_dataset_path", lambda: str(tmp_path / "datasets")
    )
    dialog = SimpleNamespace(
        image_list=images,
        output_dir=None,
        selected_task_type="Segment",
        append_training_log=lambda _message: None,
        _uses_existing_dataset=lambda _path: pytest.fail(
            "Saved YAML was inspected"
        ),
    )
    config = {
        "basic": {
            "data": str(tmp_path / "old.yaml"),
            "data_source": "auto",
            "auto_classes": ["building"],
            "dataset_ratio": 0.5,
            "model": "base.pt",
            "project": str(tmp_path / "runs"),
            "name": "exp",
            "device": "cpu",
            "pose_config": "",
        },
        "checkpoint": {},
        "train": {"classes": [4], "single_cls": True},
    }

    args = UltralyticsDialog.get_training_args(dialog, config)

    assert Path(args["data"]).is_file()
    assert load_yaml_config(args["data"])["names"] == {0: "building"}
    assert "classes" not in args
    assert "single_cls" not in args
