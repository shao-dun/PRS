"""Create reproducible train/validation/test CSV manifests for NEU-DET."""

from __future__ import annotations

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def class_images(image_root: Path, class_name: str) -> list[Path]:
    class_dir = image_root / class_name
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("path", "label", "box_count", "boxes"),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def read_boxes(annotation_path: Path) -> list[dict[str, object]]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"missing annotation: {annotation_path}")

    root = ET.parse(annotation_path).getroot()
    boxes: list[dict[str, object]] = []
    for defect in root.findall("object"):
        bounds = defect.find("bndbox")
        if bounds is None:
            raise ValueError(f"missing bndbox in {annotation_path}")
        box = {
            "label": defect.findtext("name", default=""),
            "xmin": int(bounds.findtext("xmin", default="0")),
            "ymin": int(bounds.findtext("ymin", default="0")),
            "xmax": int(bounds.findtext("xmax", default="0")),
            "ymax": int(bounds.findtext("ymax", default="0")),
        }
        if not (0 <= box["xmin"] < box["xmax"] <= 200):
            raise ValueError(f"invalid horizontal box coordinates in {annotation_path}: {box}")
        if not (0 <= box["ymin"] < box["ymax"] <= 200):
            raise ValueError(f"invalid vertical box coordinates in {annotation_path}: {box}")
        boxes.append(box)

    if not boxes:
        raise ValueError(f"no defect boxes found in {annotation_path}")
    return boxes


def make_row(path: Path, label: str, data_root: Path) -> dict[str, str]:
    source_split = path.relative_to(data_root).parts[0]
    annotation_path = data_root / source_split / "annotations" / f"{path.stem}.xml"
    boxes = read_boxes(annotation_path)
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "label": label,
        "box_count": str(len(boxes)),
        "boxes": json.dumps(boxes, ensure_ascii=False, separators=(",", ":")),
    }


def create_splits(data_root: Path, output_dir: Path, seed: int) -> None:
    train_root = data_root / "train" / "images"
    validation_root = data_root / "validation" / "images"
    if not train_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError("expected data/train/images and data/validation/images")

    random_generator = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for class_name in EXPECTED_CLASSES:
        train_images = class_images(train_root, class_name)
        candidates = class_images(validation_root, class_name)
        if len(train_images) != 240:
            raise ValueError(f"{class_name}: expected 240 training images, found {len(train_images)}")
        if len(candidates) != 60:
            raise ValueError(f"{class_name}: expected 60 validation images, found {len(candidates)}")

        random_generator.shuffle(candidates)
        test_images = sorted(candidates[:30])
        validation_images = sorted(candidates[30:])

        train_rows.extend(make_row(path, class_name, data_root) for path in train_images)
        validation_rows.extend(
            make_row(path, class_name, data_root) for path in validation_images
        )
        test_rows.extend(make_row(path, class_name, data_root) for path in test_images)

    all_paths = [row["path"] for row in train_rows + validation_rows + test_rows]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("the generated manifests contain overlapping image paths")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "train.csv", train_rows)
    write_manifest(output_dir / "val.csv", validation_rows)
    write_manifest(output_dir / "test.csv", test_rows)

    print(f"Seed: {seed}")
    for name, rows in (
        ("train", train_rows),
        ("val", validation_rows),
        ("test", test_rows),
    ):
        counts = Counter(row["label"] for row in rows)
        box_count = sum(int(row["box_count"]) for row in rows)
        per_class = ", ".join(f"{key}={counts[key]}" for key in EXPECTED_CLASSES)
        print(f"{name:>5}: {len(rows):>4} images, {box_count:>4} boxes ({per_class})")
    print(f"Saved manifests to: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "splits")
    args = parser.parse_args()
    create_splits(args.data_root.resolve(), args.output_dir.resolve(), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
