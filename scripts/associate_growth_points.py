import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
DEFAULT_BOX_LABELS = ["broadleaf_weed", "grass_weed", "sedge_weed"]


@dataclass
class BoxInstance:
    index: int
    label: str
    points: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]
    center: Tuple[float, float]


@dataclass
class PointInstance:
    index: int
    label: str
    point: Tuple[float, float]


@dataclass
class CandidateMatch:
    box_index: int
    point_index: int
    distance: float
    match_type: str
    had_multi_points: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Associate rotation boxes with growth points in Labelme-compatible JSON files."
    )
    parser.add_argument("--image-dir", default="dataset/image")
    parser.add_argument("--json-dir", default="dataset/json")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--box-labels", nargs="+", default=DEFAULT_BOX_LABELS)
    parser.add_argument("--point-label", default="gp")
    parser.add_argument("--expand-ratio", type=float, default=0.1)
    parser.add_argument("--vis-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-yolo-pose", action="store_true")
    parser.add_argument("--label-out-dir", default="output/yolo_pose_labels")
    return parser.parse_args()


def ensure_image_dir(image_dir: Path) -> Tuple[Path, Optional[str]]:
    if image_dir.exists():
        return image_dir, None

    alt_dir = image_dir.parent / f"{image_dir.name}s"
    if alt_dir.exists():
        warning = (
            f"image directory '{image_dir}' not found, fallback to '{alt_dir}'"
        )
        return alt_dir, warning

    return image_dir, f"image directory '{image_dir}' not found"


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clip01(value: float) -> float:
    return clip_value(value, 0.0, 1.0)


def clip_point(point: Sequence[float], width: int, height: int) -> Tuple[float, float]:
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    x = clip_value(float(point[0]), 0.0, float(max_x))
    y = clip_value(float(point[1]), 0.0, float(max_y))
    return x, y


def quad_to_bbox(points: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    x_min = clip_value(float(np.min(points[:, 0])), 0.0, float(max_x))
    y_min = clip_value(float(np.min(points[:, 1])), 0.0, float(max_y))
    x_max = clip_value(float(np.max(points[:, 0])), 0.0, float(max_x))
    y_max = clip_value(float(np.max(points[:, 1])), 0.0, float(max_y))
    return x_min, y_min, x_max, y_max


def expand_bbox(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    expand_ratio: float,
) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    box_w = max(x_max - x_min, 0.0)
    box_h = max(y_max - y_min, 0.0)
    expand_x = box_w * expand_ratio
    expand_y = box_h * expand_ratio
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    return (
        clip_value(x_min - expand_x, 0.0, float(max_x)),
        clip_value(y_min - expand_y, 0.0, float(max_y)),
        clip_value(x_max + expand_x, 0.0, float(max_x)),
        clip_value(y_max + expand_y, 0.0, float(max_y)),
    )


def bbox_contains_point(bbox: Tuple[float, float, float, float], point: Tuple[float, float]) -> bool:
    x_min, y_min, x_max, y_max = bbox
    x, y = point
    return x_min <= x <= x_max and y_min <= y <= y_max


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def resolve_image_path(image_dir: Path, image_name: str) -> Optional[Path]:
    candidate = image_dir / image_name
    if candidate.exists():
        return candidate

    stem = Path(image_name).stem
    for ext in IMAGE_EXTENSIONS:
        alt = image_dir / f"{stem}{ext}"
        if alt.exists():
            return alt
    return None


def load_instances(
    data: Dict,
    box_labels: Sequence[str],
    point_label: str,
) -> Tuple[int, int, str, List[BoxInstance], List[PointInstance]]:
    image_width = int(data.get("imageWidth", 0) or 0)
    image_height = int(data.get("imageHeight", 0) or 0)
    image_name = data.get("imagePath", "")
    box_label_set = set(box_labels)

    boxes: List[BoxInstance] = []
    points: List[PointInstance] = []

    for shape in data.get("shapes", []) or []:
        label = shape.get("label")
        shape_type = shape.get("shape_type")
        raw_points = shape.get("points", []) or []

        if shape_type == "rotation" and label in box_label_set and len(raw_points) >= 4:
            quad = np.array(
                [clip_point(point, image_width, image_height) for point in raw_points[:4]],
                dtype=np.float32,
            )
            bbox = quad_to_bbox(quad, image_width, image_height)
            center = bbox_center(bbox)
            boxes.append(
                BoxInstance(
                    index=len(boxes),
                    label=label,
                    points=quad,
                    bbox_xyxy=bbox,
                    center=center,
                )
            )
        elif shape_type == "point" and label == point_label and len(raw_points) >= 1:
            point = clip_point(raw_points[0], image_width, image_height)
            points.append(
                PointInstance(
                    index=len(points),
                    label=label,
                    point=point,
                )
            )

    return image_width, image_height, image_name, boxes, points


def build_candidates(
    boxes: Sequence[BoxInstance],
    points: Sequence[PointInstance],
    width: int,
    height: int,
    expand_ratio: float,
) -> Tuple[List[CandidateMatch], Dict[int, Dict], int, int]:
    candidates: List[CandidateMatch] = []
    preliminary_info: Dict[int, Dict] = {}
    multi_point_boxes = 0
    expanded_matches = 0

    for box in boxes:
        inside_points = [
            point for point in points if bbox_contains_point(box.bbox_xyxy, point.point)
        ]
        expanded_points: List[PointInstance] = []

        if len(inside_points) == 1:
            point = inside_points[0]
            candidates.append(
                CandidateMatch(
                    box_index=box.index,
                    point_index=point.index,
                    distance=euclidean_distance(box.center, point.point),
                    match_type="inside_single",
                    had_multi_points=False,
                )
            )
            preliminary_info[box.index] = {
                "status": "inside_single",
                "num_candidate_points_inside": len(inside_points),
                "num_candidate_points_expanded": 0,
            }
            continue

        if len(inside_points) > 1:
            multi_point_boxes += 1
            point = min(
                inside_points,
                key=lambda item: euclidean_distance(box.center, item.point),
            )
            candidates.append(
                CandidateMatch(
                    box_index=box.index,
                    point_index=point.index,
                    distance=euclidean_distance(box.center, point.point),
                    match_type="inside_nearest",
                    had_multi_points=True,
                )
            )
            preliminary_info[box.index] = {
                "status": "inside_nearest",
                "num_candidate_points_inside": len(inside_points),
                "num_candidate_points_expanded": 0,
            }
            continue

        expanded_bbox = expand_bbox(box.bbox_xyxy, width, height, expand_ratio)
        expanded_points = [
            point for point in points if bbox_contains_point(expanded_bbox, point.point)
        ]
        if expanded_points:
            expanded_matches += 1
            point = min(
                expanded_points,
                key=lambda item: euclidean_distance(box.center, item.point),
            )
            candidates.append(
                CandidateMatch(
                    box_index=box.index,
                    point_index=point.index,
                    distance=euclidean_distance(box.center, point.point),
                    match_type="expanded_nearest",
                    had_multi_points=len(expanded_points) > 1,
                )
            )
            preliminary_info[box.index] = {
                "status": "expanded_nearest",
                "num_candidate_points_inside": 0,
                "num_candidate_points_expanded": len(expanded_points),
            }
        else:
            preliminary_info[box.index] = {
                "status": "unmatched",
                "num_candidate_points_inside": 0,
                "num_candidate_points_expanded": 0,
            }

    return candidates, preliminary_info, multi_point_boxes, expanded_matches


def resolve_conflicts(
    boxes: Sequence[BoxInstance],
    points: Sequence[PointInstance],
    candidates: Sequence[CandidateMatch],
    preliminary_info: Dict[int, Dict],
) -> Tuple[Dict[int, Dict], int]:
    results: Dict[int, Dict] = {
        box.index: {
            "status": preliminary_info.get(box.index, {}).get("status", "unmatched"),
            "label": box.label,
            "point_index": None,
            "distance": None,
            "center": box.center,
            "matched_point": None,
            "num_candidate_points_inside": preliminary_info.get(box.index, {}).get(
                "num_candidate_points_inside", 0
            ),
            "num_candidate_points_expanded": preliminary_info.get(box.index, {}).get(
                "num_candidate_points_expanded", 0
            ),
        }
        for box in boxes
    }

    by_point: Dict[int, List[CandidateMatch]] = {}
    for candidate in candidates:
        by_point.setdefault(candidate.point_index, []).append(candidate)

    conflict_unmatched = 0
    point_lookup = {point.index: point for point in points}

    for point_index, point_candidates in by_point.items():
        point_candidates = sorted(
            point_candidates,
            key=lambda item: (item.distance, item.box_index),
        )
        winner = point_candidates[0]
        point = point_lookup[point_index]
        results[winner.box_index].update(
            {
                "status": winner.match_type,
                "point_index": point_index,
                "distance": winner.distance,
                "matched_point": point.point,
            }
        )

        for loser in point_candidates[1:]:
            results[loser.box_index].update(
                {
                    "status": "conflict_unmatched",
                    "point_index": None,
                    "distance": None,
                    "matched_point": None,
                }
            )
            conflict_unmatched += 1

    return results, conflict_unmatched


def draw_visualization(
    image: np.ndarray,
    boxes: Sequence[BoxInstance],
    points: Sequence[PointInstance],
    match_results: Dict[int, Dict],
    out_path: Path,
) -> None:
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    blue = (255, 0, 0)

    canvas = image.copy()

    for point in points:
        x, y = point.point
        cv2.circle(canvas, (int(round(x)), int(round(y))), 4, red, -1)

    for box in boxes:
        result = match_results[box.index]
        status = result["status"]
        color = green if status in {"inside_single", "inside_nearest", "expanded_nearest"} else yellow

        cv2.polylines(
            canvas,
            [np.round(box.points).astype(np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=color,
            thickness=2,
        )

        center_x, center_y = box.center
        if result["matched_point"] is not None:
            point_x, point_y = result["matched_point"]
            cv2.line(
                canvas,
                (int(round(center_x)), int(round(center_y))),
                (int(round(point_x)), int(round(point_y))),
                blue,
                2,
            )

        text = f"{box.label} {status}"
        x_min, y_min, _, _ = box.bbox_xyxy
        text_origin = (int(round(x_min)), max(int(round(y_min)) - 6, 12))
        cv2.putText(
            canvas,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    args = parse_args()

    json_dir = Path(args.json_dir)
    image_dir, image_dir_warning = ensure_image_dir(Path(args.image_dir))
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis_check"
    report_path = out_dir / "association_report.csv"
    summary_path = out_dir / "association_summary.json"
    instances_path = out_dir / "association_instances.csv"
    yolo_pose_summary_path = out_dir / "yolo_pose_export_summary.json"
    label_out_dir = Path(args.label_out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if args.export_yolo_pose:
        label_out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    if image_dir_warning:
        warnings.append(image_dir_warning)

    if not json_dir.exists():
        raise FileNotFoundError(f"json directory not found: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))
    rng = random.Random(args.seed)
    vis_names = set(
        path.name
        for path in rng.sample(json_files, min(args.vis_num, len(json_files)))
    )

    rows: List[Dict] = []
    instance_rows: List[Dict] = []
    total_files = len(json_files)
    total_boxes = 0
    total_points = 0
    total_matched = 0
    total_unmatched_boxes = 0
    total_unused_points = 0
    total_multi_point_boxes = 0
    total_expanded_matches = 0
    total_conflict_unmatched = 0
    yolo_files_with_labels = 0
    yolo_files_without_labels = 0
    yolo_total_exported_instances = 0
    yolo_skipped_unmatched = 0
    yolo_skipped_conflict_unmatched = 0

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"failed to read {json_path.name}: {exc}")
            continue

        width, height, image_name, boxes, points = load_instances(
            data=data,
            box_labels=args.box_labels,
            point_label=args.point_label,
        )

        candidates, preliminary_info, multi_point_boxes, expanded_matches = build_candidates(
            boxes=boxes,
            points=points,
            width=width,
            height=height,
            expand_ratio=args.expand_ratio,
        )
        match_results, conflict_unmatched = resolve_conflicts(
            boxes=boxes,
            points=points,
            candidates=candidates,
            preliminary_info=preliminary_info,
        )

        matched_point_indices = {
            result["point_index"]
            for result in match_results.values()
            if result["point_index"] is not None
        }
        num_matched = len(matched_point_indices)
        num_boxes = len(boxes)
        num_points = len(points)
        num_unmatched_boxes = sum(
            1
            for result in match_results.values()
            if result["status"] in {"unmatched", "conflict_unmatched"}
        )
        num_unused_points = num_points - num_matched

        row = {
            "json_name": json_path.name,
            "image_name": image_name,
            "image_width": width,
            "image_height": height,
            "num_boxes": num_boxes,
            "num_points": num_points,
            "num_matched": num_matched,
            "num_unmatched_boxes": num_unmatched_boxes,
            "num_unused_points": num_unused_points,
            "num_multi_point_boxes": multi_point_boxes,
            "num_expanded_matches": sum(
                1 for result in match_results.values() if result["status"] == "expanded_nearest"
            ),
            "num_conflict_unmatched": conflict_unmatched,
        }
        rows.append(row)

        for box in boxes:
            result = match_results[box.index]
            x1, y1, x2, y2 = box.bbox_xyxy
            cx, cy = box.center
            matched = 1 if result["point_index"] is not None else 0
            instance_row = {
                "json_name": json_path.name,
                "image_name": image_name,
                "box_id": box.index,
                "box_label": box.label,
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "box_cx": cx,
                "box_cy": cy,
                "matched": matched,
                "match_type": result["status"],
                "point_id": "" if not matched else result["point_index"],
                "point_x": "" if not matched else result["matched_point"][0],
                "point_y": "" if not matched else result["matched_point"][1],
                "distance_to_center": "" if not matched else result["distance"],
                "num_candidate_points_inside": result["num_candidate_points_inside"],
                "num_candidate_points_expanded": result["num_candidate_points_expanded"],
            }
            instance_rows.append(instance_row)

        total_boxes += num_boxes
        total_points += num_points
        total_matched += num_matched
        total_unmatched_boxes += num_unmatched_boxes
        total_unused_points += num_unused_points
        total_multi_point_boxes += multi_point_boxes
        total_expanded_matches += row["num_expanded_matches"]
        total_conflict_unmatched += conflict_unmatched

        if args.export_yolo_pose:
            yolo_lines: List[str] = []
            for box in boxes:
                result = match_results[box.index]
                status = result["status"]
                if status == "conflict_unmatched":
                    yolo_skipped_conflict_unmatched += 1
                    continue
                if result["point_index"] is None:
                    yolo_skipped_unmatched += 1
                    continue

                x1, y1, x2, y2 = box.bbox_xyxy
                box_w = max(x2 - x1, 0.0)
                box_h = max(y2 - y1, 0.0)
                box_cx, box_cy = box.center
                point_x, point_y = result["matched_point"]

                norm_x_center = clip01(box_cx / width) if width else 0.0
                norm_y_center = clip01(box_cy / height) if height else 0.0
                norm_width = clip01(box_w / width) if width else 0.0
                norm_height = clip01(box_h / height) if height else 0.0
                norm_kpt_x = clip01(point_x / width) if width else 0.0
                norm_kpt_y = clip01(point_y / height) if height else 0.0

                yolo_lines.append(
                    " ".join(
                        [
                            "0",
                            f"{norm_x_center:.6f}",
                            f"{norm_y_center:.6f}",
                            f"{norm_width:.6f}",
                            f"{norm_height:.6f}",
                            f"{norm_kpt_x:.6f}",
                            f"{norm_kpt_y:.6f}",
                            "2",
                        ]
                    )
                )

            txt_name = f"{json_path.stem}.txt"
            txt_path = label_out_dir / txt_name
            if yolo_lines:
                txt_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
                yolo_files_with_labels += 1
                yolo_total_exported_instances += len(yolo_lines)
            else:
                txt_path.write_text("", encoding="utf-8")
                yolo_files_without_labels += 1

        if json_path.name in vis_names:
            image_path = resolve_image_path(image_dir, image_name)
            if image_path is None:
                warnings.append(
                    f"image not found for {json_path.name}: expected '{image_name}' under '{image_dir}'"
                )
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    warnings.append(f"failed to load image for visualization: {image_path}")
                else:
                    vis_path = vis_dir / image_path.name
                    draw_visualization(
                        image=image,
                        boxes=boxes,
                        points=points,
                        match_results=match_results,
                        out_path=vis_path,
                    )

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "json_name",
                "image_name",
                "image_width",
                "image_height",
                "num_boxes",
                "num_points",
                "num_matched",
                "num_unmatched_boxes",
                "num_unused_points",
                "num_multi_point_boxes",
                "num_expanded_matches",
                "num_conflict_unmatched",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with instances_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "json_name",
                "image_name",
                "box_id",
                "box_label",
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
                "box_cx",
                "box_cy",
                "matched",
                "match_type",
                "point_id",
                "point_x",
                "point_y",
                "distance_to_center",
                "num_candidate_points_inside",
                "num_candidate_points_expanded",
            ],
        )
        writer.writeheader()
        writer.writerows(instance_rows)

    match_rate = (total_matched / total_boxes) if total_boxes else 0.0
    summary = {
        "total_files": total_files,
        "total_boxes": total_boxes,
        "total_points": total_points,
        "total_matched": total_matched,
        "total_unmatched_boxes": total_unmatched_boxes,
        "total_unused_points": total_unused_points,
        "total_multi_point_boxes": total_multi_point_boxes,
        "total_expanded_matches": total_expanded_matches,
        "total_conflict_unmatched": total_conflict_unmatched,
        "match_rate": match_rate,
        "box_labels": list(args.box_labels),
        "point_label": args.point_label,
        "expand_ratio": args.expand_ratio,
        "image_dir": str(image_dir),
        "json_dir": str(json_dir),
        "warnings": warnings,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    yolo_pose_summary = None
    if args.export_yolo_pose:
        yolo_pose_summary = {
            "total_files": total_files,
            "files_with_labels": yolo_files_with_labels,
            "files_without_labels": yolo_files_without_labels,
            "total_exported_instances": yolo_total_exported_instances,
            "skipped_unmatched": yolo_skipped_unmatched,
            "skipped_conflict_unmatched": yolo_skipped_conflict_unmatched,
            "class_names": ["weed"],
            "kpt_shape": [1, 3],
            "label_out_dir": str(label_out_dir),
        }
        yolo_pose_summary_path.write_text(
            json.dumps(yolo_pose_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("Association summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"report_csv: {report_path}")
    print(f"instances_csv: {instances_path}")
    print(f"summary_json: {summary_path}")
    print(f"vis_dir: {vis_dir}")
    if yolo_pose_summary is not None:
        print("YOLO pose export summary:")
        print(json.dumps(yolo_pose_summary, indent=2, ensure_ascii=False))
        print(f"yolo_pose_summary_json: {yolo_pose_summary_path}")
        print(f"yolo_pose_label_dir: {label_out_dir}")


if __name__ == "__main__":
    main()
