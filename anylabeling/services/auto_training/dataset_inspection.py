"""Read labeled images without imposing a training framework's file format."""

import json
import os
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DatasetInspection:
    annotation_counts: Counter = field(default_factory=Counter)
    image_counts: Counter = field(default_factory=Counter)
    image_labels: dict = field(default_factory=dict)
    missing_images: int = 0
    missing_annotations: int = 0
    invalid_annotations: int = 0
    unchecked_annotations: int = 0

    def images_with_any(self, class_names):
        selected = set(class_names)
        return sum(
            bool(labels & selected) for labels in self.image_labels.values()
        )


def is_usable_shape(shape, allowed_shapes):
    if (
        not isinstance(shape, dict)
        or shape.get("shape_type") not in allowed_shapes
    ):
        return False
    if not isinstance(shape.get("label"), str) or not shape["label"].strip():
        return False
    points = shape.get("points")
    if shape["shape_type"] == "polygon":
        return isinstance(points, list) and len(points) >= 3
    if shape["shape_type"] == "rotation":
        return isinstance(points, list) and len(points) == 4
    if shape["shape_type"] == "rectangle":
        return isinstance(points, list) and len(points) in (2, 4)
    return True


def inspect_annotations(
    image_paths,
    shape_types,
    output_dir=None,
    only_checked=False,
):
    """Count labels matching the selected task in image/JSON pairs."""
    result = DatasetInspection()
    allowed_shapes = set(shape_types)

    for image_path in image_paths:
        if not os.path.isfile(image_path):
            result.missing_images += 1
            continue
        label_dir = output_dir or os.path.dirname(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, stem + ".json")
        if not os.path.isfile(label_path):
            result.missing_annotations += 1
            continue
        try:
            with open(label_path, "r", encoding="utf-8") as stream:
                annotation = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError):
            result.invalid_annotations += 1
            continue
        if not isinstance(annotation, dict):
            result.invalid_annotations += 1
            continue
        if only_checked and annotation.get("checked") is not True:
            result.unchecked_annotations += 1
            continue

        if "flags" in allowed_shapes:
            flags = annotation.get("flags")
            if not isinstance(flags, dict):
                result.invalid_annotations += 1
                continue
            labels = [name for name, active in flags.items() if active]
        else:
            shapes = annotation.get("shapes")
            if not isinstance(shapes, list):
                result.invalid_annotations += 1
                continue
            if (
                not isinstance(annotation.get("imageWidth"), int)
                or not isinstance(annotation.get("imageHeight"), int)
                or annotation["imageWidth"] <= 0
                or annotation["imageHeight"] <= 0
            ):
                result.invalid_annotations += 1
                continue
            labels = []
            for shape in shapes:
                if is_usable_shape(shape, allowed_shapes):
                    labels.append(shape["label"])

        if not labels:
            continue
        result.image_labels[image_path] = set(labels)
        result.annotation_counts.update(labels)
        result.image_counts.update(set(labels))

    return result
