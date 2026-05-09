# YOLO Weed Detection

This repository is organized for two goals:

- keep the key project assets needed to continue the work on another machine
- give Codex enough stable project context to resume effectively

## Start Here

- `AGENTS.md`: repo rules and working context for Codex
- `scripts/`: dataset conversion and split scripts
- `custom_models/`: custom YOLO pose model definitions
- `dataset/weed_pose/`: dataset config, split summary, and split labels
- `experiments/`: compact artifacts from the main training runs
- `output/`: preprocessing summaries and risk analysis tables

## What Is Kept

- reproducible preprocessing scripts
- custom model YAML files
- split label files for train / val / test
- dataset metadata
- key experiment outputs
- lightweight project summaries for future collaboration

## What Is Not Kept

- raw tiled images
- original Labelme JSON annotation files
- IDE settings
- duplicate exported label folders
- large visualization samples that are not required for reproduction

The raw image set is intentionally excluded because it is large and better stored separately or with Git LFS / cloud storage.

## Main Experiments

| Experiment | Box mAP50-95 | Pose mAP50-95 |
| --- | ---: | ---: |
| `yolo11n_pose_200` | `0.58993` | `0.90574` |
| `yolo11n_pose_eca_200` | `0.51979` | `0.88316` |
| `yolo11n_pose_p2_200` | `0.55166` | `0.89608` |
| `yolov8n_pose_200` | `0.57201` | `0.89318` |

## Dataset Summary

- labeled images exported to YOLO pose format: `1208`
- train / val / test: `966 / 120 / 122`
- skipped empty labels during split: `332`
- exported matched weed instances: `6904`
- box-point association match rate: `0.9090`

## Reproduction

Generate YOLO pose labels:

```bash
python scripts/associate_growth_points.py --image-dir dataset/images --json-dir dataset/json --out-dir output --export-yolo-pose --label-out-dir output/yolo_pose_labels
```

Split the dataset:

```bash
python scripts/split_pose_dataset.py --image-dir dataset/images --label-dir output/yolo_pose_labels --out-dir dataset/weed_pose
```

Train with Ultralytics YOLO pose:

```bash
yolo pose train model=yolo11n-pose.pt data=dataset/weed_pose/weed_pose.yaml epochs=200 imgsz=640
```

Use a custom model by replacing `model=` with a YAML file from `custom_models/`.
