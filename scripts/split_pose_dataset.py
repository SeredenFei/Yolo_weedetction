import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split YOLO pose dataset into train/val/test directories."
    )
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--label-dir", default="output/yolo_pose_labels")
    parser.add_argument("--out-dir", default="dataset/weed_pose")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("sum of split ratios must be greater than 0")

    return tuple(ratio / total for ratio in ratios)


def resolve_image_path(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def is_empty_label_file(label_path: Path) -> bool:
    try:
        return label_path.read_text(encoding="utf-8").strip() == ""
    except UnicodeDecodeError:
        return label_path.read_text().strip() == ""


def collect_labeled_pairs(image_dir: Path, label_dir: Path) -> Tuple[List[Dict[str, Path]], int, int]:
    pairs: List[Dict[str, Path]] = []
    skipped_empty_labels = 0
    skipped_missing_images = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        if is_empty_label_file(label_path):
            skipped_empty_labels += 1
            continue

        image_path = resolve_image_path(image_dir, label_path.stem)
        if image_path is None:
            skipped_missing_images += 1
            continue

        pairs.append(
            {
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    return pairs, skipped_empty_labels, skipped_missing_images


def compute_split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if test_count < 0:
        raise ValueError("invalid split counts computed from ratios")

    return train_count, val_count, test_count


def ensure_output_dirs(out_dir: Path) -> Dict[str, Dict[str, Path]]:
    dirs: Dict[str, Dict[str, Path]] = {
        "images": {},
        "labels": {},
    }
    for group in ("images", "labels"):
        for split in ("train", "val", "test"):
            split_dir = out_dir / group / split
            split_dir.mkdir(parents=True, exist_ok=True)
            dirs[group][split] = split_dir
    return dirs


def copy_split_files(
    items: Sequence[Dict[str, Path]],
    split_name: str,
    out_dirs: Dict[str, Dict[str, Path]],
) -> None:
    image_out_dir = out_dirs["images"][split_name]
    label_out_dir = out_dirs["labels"][split_name]

    for item in items:
        image_path = item["image_path"]
        label_path = item["label_path"]
        shutil.copy2(image_path, image_out_dir / image_path.name)
        shutil.copy2(label_path, label_out_dir / label_path.name)


def write_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "weed_pose.yaml"
    yaml_content = "\n".join(
        [
            f"path: {out_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: weed",
            "",
            "kpt_shape: [1, 3]",
            "",
        ]
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label directory not found: {label_dir}")

    train_ratio, val_ratio, test_ratio = validate_ratios(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    pairs, skipped_empty_labels, skipped_missing_images = collect_labeled_pairs(
        image_dir=image_dir,
        label_dir=label_dir,
    )

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    total_labeled_images = len(pairs)
    train_count, val_count, test_count = compute_split_counts(
        total=total_labeled_images,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_items = pairs[:train_count]
    val_items = pairs[train_count : train_count + val_count]
    test_items = pairs[train_count + val_count :]

    out_dirs = ensure_output_dirs(out_dir)
    copy_split_files(train_items, "train", out_dirs)
    copy_split_files(val_items, "val", out_dirs)
    copy_split_files(test_items, "test", out_dirs)

    yaml_path = write_yaml(out_dir)
    summary = {
        "total_labeled_images": total_labeled_images,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "skipped_empty_labels": skipped_empty_labels,
        "skipped_missing_images": skipped_missing_images,
        "out_dir": str(out_dir),
    }

    summary_path = out_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Split summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary_json: {summary_path}")
    print(f"yaml_path: {yaml_path}")


if __name__ == "__main__":
    main()
