"""Quick integrity check for the NEU surface-defect dataset."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
EXPECTED_CLASSES = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> bool:
    images: list[tuple[str, str, Path]] = []
    counts: Counter[tuple[str, str]] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    hashes: defaultdict[str, list[Path]] = defaultdict(list)
    broken: list[tuple[Path, str]] = []

    for split in ("train", "validation"):
        image_root = root / split / "images"
        if not image_root.is_dir():
            print(f"ERROR: missing directory: {image_root}")
            return False

        for path in sorted(image_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            class_name = path.parent.name
            images.append((split, class_name, path))
            counts[split, class_name] += 1
            hashes[sha256(path)].append(path)

            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    sizes[image.size] += 1
                    modes[image.mode] += 1
                    image.load()
            except Exception as exc:  # Pillow reports format-specific errors.
                broken.append((path, str(exc)))

    classes = {class_name for _, class_name, _ in images}
    class_totals = Counter(class_name for _, class_name, _ in images)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]

    print("\nImage counts")
    print(f"{'class':<20} {'train':>8} {'validation':>12} {'total':>8}")
    for class_name in sorted(classes | EXPECTED_CLASSES):
        print(
            f"{class_name:<20} {counts['train', class_name]:>8} "
            f"{counts['validation', class_name]:>12} {class_totals[class_name]:>8}"
        )
    print(f"{'TOTAL':<20} {sum(v for (s, _), v in counts.items() if s == 'train'):>8} "
          f"{sum(v for (s, _), v in counts.items() if s == 'validation'):>12} "
          f"{len(images):>8}")

    print(f"\nImage sizes: {dict(sizes)}")
    print(f"Image modes: {dict(modes)}")
    print(f"Unreadable images: {len(broken)}")
    for path, reason in broken:
        print(f"  - {path.relative_to(root)}: {reason}")

    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    for group in duplicate_groups:
        print("  - " + " == ".join(str(path.relative_to(root)) for path in group))

    # XML annotations are optional for classification, but report misplaced files.
    image_split_by_stem = {
        (class_name, path.stem): split for split, class_name, path in images
    }
    misplaced_xml: list[tuple[Path, str]] = []
    xml_count = 0
    for split in ("train", "validation"):
        annotation_root = root / split / "annotations"
        if not annotation_root.is_dir():
            continue
        for xml_path in annotation_root.glob("*.xml"):
            xml_count += 1
            class_name = xml_path.stem.rsplit("_", 1)[0]
            image_split = image_split_by_stem.get((class_name, xml_path.stem))
            if image_split is not None and image_split != split:
                misplaced_xml.append((xml_path, image_split))

    print(f"XML annotations: {xml_count}")
    print(f"Misplaced XML files: {len(misplaced_xml)}")
    for path, expected_split in misplaced_xml:
        print(f"  - {path.relative_to(root)} (image is in {expected_split})")

    complete = (
        len(images) == 1800
        and classes == EXPECTED_CLASSES
        and all(class_totals[name] == 300 for name in EXPECTED_CLASSES)
        and not broken
    )
    status = "PASS" if complete else "FAIL"
    if complete and (duplicate_groups or misplaced_xml):
        status = "PASS WITH WARNINGS"
    print(f"\nDataset check: {status}")
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    args = parser.parse_args()
    return 0 if audit(args.root.resolve()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
