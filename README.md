# YOLO Weed Detection

This repository contains the key assets from the weed detection and growth-point keypoint project:

- dataset preprocessing scripts
- custom YOLO pose model configs
- exported YOLO pose labels and split metadata
- compact training artifacts for the main experiments
- selected preprocessing statistics and visualization samples

## Included

- `scripts/associate_growth_points.py`
- `scripts/split_pose_dataset.py`
- `custom_models/`
- `dataset/weed_pose/weed_pose.yaml`
- `dataset/weed_pose/split_summary.json`
- `dataset/weed_pose/labels/`
- `output/association_summary.json`
- `output/association_report.csv`
- `output/association_instances.csv`
- `output/association_risk_samples.csv`
- `output/yolo_pose_export_summary.json`
- `output/yolo_pose_labels/`
- `output/vis_check_samples/`
- `experiments/`

## Not Included

The raw tiled images and original Labelme JSON annotations are not committed here because the image set is about 1.95 GB and would make the repository unnecessarily heavy. If needed, they should be distributed separately or moved to Git LFS / cloud storage.

## Dataset Summary

- labeled images exported to YOLO pose format: `1208`
- train / val / test: `966 / 120 / 122`
- skipped empty labels during split: `332`
- exported matched weed instances: `6904`
- box-point association match rate: `0.9090`

## Main Experiments

Compact artifacts are stored under `experiments/`. Each experiment keeps:

- `args.yaml`
- `results.csv`
- summary curves and confusion matrices
- `weights/best.pt`

Final epoch metrics from the main runs:

| Experiment | Box mAP50-95 | Pose mAP50-95 |
| --- | ---: | ---: |
| `yolo11n_pose_200` | `0.58993` | `0.90574` |
| `yolo11n_pose_eca_200` | `0.51979` | `0.88316` |
| `yolo11n_pose_p2_200` | `0.55166` | `0.89608` |
| `yolov8n_pose_200` | `0.57201` | `0.89318` |

## Reproduction

1. Generate YOLO pose labels from Labelme-style annotations:

```bash
python scripts/associate_growth_points.py --image-dir dataset/images --json-dir dataset/json --out-dir output --export-yolo-pose --label-out-dir output/yolo_pose_labels
```

2. Split the exported labels into train / val / test:

```bash
python scripts/split_pose_dataset.py --image-dir dataset/images --label-dir output/yolo_pose_labels --out-dir dataset/weed_pose
```

3. Train with Ultralytics YOLO pose:

```bash
yolo pose train model=yolo11n-pose.pt data=dataset/weed_pose/weed_pose.yaml epochs=200 imgsz=640
```

For custom structures, replace `model=` with one of the YAML files under `custom_models/`.
