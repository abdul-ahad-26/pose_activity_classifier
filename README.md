# Pose-Based Human Activity Classifier

Computer Vision — Complex Computing Problem (CLO-3). A pipeline that detects human
body keypoints with a pre-trained pose model, smooths them, computes joint angles,
and classifies activity (**standing** vs **sitting**) per frame with a rule-based
classifier, reporting accuracy against a manually-labeled ground truth.

## Video source
Own **pre-recorded clip**: [`activity.mp4`](activity.mp4) — ~20.5 s, 613 frames @ 30 fps,
one person repeatedly standing up and sitting down.

## Tasks covered
1. **Pose detection & pre-processing** — MediaPipe Pose (33 landmarks/frame), missing
   detections interpolated, **Savitzky-Golay** smoothing of keypoint trajectories,
   skeleton overlay video.
2. **Joint angle computation & tracking** — knee, hip, and elbow angles via the
   `arccos` dot-product formula; angles plotted over time with transition frames marked.
3. **Rule-based classification** — `standing` if `knee > 130°` **and** `hip > 130°`,
   else `sitting`; per-frame accuracy vs. `ground_truth.csv`.

## Result
**99.35% accuracy** (609/613 frames). All 4 errors fall on stand-up/sit-down
transition frames, where the posture is genuinely ambiguous.

## Setup & run (Windows, uv + Python 3.12)
MediaPipe ships wheels only up to Python 3.12, so use a 3.12 environment:

```powershell
python -m pip install uv
uv python install 3.12
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m ipykernel install --user --name pose312 --display-name "Python 3.12 (pose)"

# run the notebook end-to-end
uv run jupyter nbconvert --to notebook --execute --inplace pose_activity_classifier.ipynb
```

Or open `pose_activity_classifier.ipynb` in Jupyter/VS Code and select the
**Python 3.12 (pose)** kernel.

## Files
| File | Description |
|------|-------------|
| `pose_activity_classifier.ipynb` | Main notebook — all three tasks, explained step by step |
| `ground_truth.csv` | Manually-labeled per-frame `standing`/`sitting` labels |
| `results.csv` | Per-frame angles + predicted + ground-truth labels |
| `report.pdf` | 1–2 page report (approach, rule logic, accuracy, observations, screenshots) |
| `outputs/annotated.mp4` | Skeleton-overlay video |
| `outputs/*.png` | Smoothing, angle, timeline, and overlay screenshots |
| `requirements.txt` | Pinned dependencies |
