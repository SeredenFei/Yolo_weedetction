# Repository Agent Rules

This repository is a compact working copy of a weed detection and growth-point keypoint project.

## Primary Goal

When working here, optimize for continuity across machines:

- preserve project context that helps future Codex sessions
- keep the repository lightweight enough to clone quickly
- avoid reintroducing raw datasets or local IDE noise

## Project Scope

- task: weed detection with YOLO pose
- object class setup: single class `weed`
- keypoint setup: one growth-point keypoint per instance
- dataset format in this repo: YOLO pose labels with train / val / test split

## Important Paths

- `scripts/associate_growth_points.py`: convert Labelme-style box + point annotations into YOLO pose labels
- `scripts/split_pose_dataset.py`: split exported labels into train / val / test
- `custom_models/`: custom model definitions such as ECA and P2 variants
- `dataset/weed_pose/weed_pose.yaml`: training dataset config
- `dataset/weed_pose/split_summary.json`: split statistics
- `experiments/`: compact artifacts from the main experiments
- `output/association_summary.json`: high-level annotation association result
- `output/association_report.csv`: per-image summary table
- `output/association_risk_samples.csv`: likely problematic samples

## Keep

- scripts
- config files
- dataset metadata
- split label files
- experiment summaries
- concise documentation that helps future sessions resume

## Avoid Adding

- `.idea/`, `.vscode/`, or local editor state
- raw image datasets unless explicitly requested
- original JSON annotations unless explicitly requested
- duplicate generated label folders
- bulky visual debug exports unless they are directly needed

## Working Style

- prefer updating `README.md` or this file when adding new project conventions
- if a future change adds a new experiment, also add a short summary entry to `README.md`
- if raw data must be referenced, document where it lives rather than committing it by default
