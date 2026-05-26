"""Build a 1-2 page report.pdf from the pipeline results (results.csv, outputs/*.png)."""
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT = "outputs"
res = pd.read_csv("results.csv")
acc = accuracy_score(res["ground_truth"], res["pred"])
labels = ["sitting", "standing"]
cm = confusion_matrix(res["ground_truth"], res["pred"], labels=labels)
mism = res["frame"][res["ground_truth"].values != res["pred"].values].tolist()

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
NEXT = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def heading(t):
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, t, **NEXT)
    pdf.set_font("Helvetica", "", 10)


def para(t):
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 5, t)


pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Pose-Based Human Activity Classifier", **NEXT)
pdf.set_font("Helvetica", "", 10)
para("Computer Vision - Complex Computing Problem (CLO-3)\n"
     "Video source: own pre-recorded clip 'activity.mp4' (~20.5 s, 613 frames @ 30 fps); "
     "a single person alternating between standing and sitting.")

heading("1. Angle Computation Approach")
para("Body keypoints were extracted from every frame with the pre-trained MediaPipe Pose model "
     "(33 landmarks per frame). Missing detections were linearly interpolated and the keypoint "
     "trajectories were de-jittered with a Savitzky-Golay filter (window 11, order 3) along the "
     "time axis. For three landmarks A-B-C, the joint angle at B is the angle between vectors B->A "
     "and B->C, computed as arccos((BA . BC)/(|BA||BC|)) in pixel coordinates. Three angles were "
     "tracked, averaging left and right sides: knee (hip-knee-ankle), hip (shoulder-hip-knee) "
     "and elbow (shoulder-elbow-wrist).")

heading("2. Rule-Based Logic")
para("Standing shows an extended knee (~150-180 deg) and an open hip (~150-180 deg); sitting shows "
     "a bent knee (~65-95 deg) and a folded hip. The classifier labels a frame as STANDING if "
     "(knee angle > 130 deg) AND (hip angle > 130 deg), otherwise SITTING. Using two joints rather "
     "than the knee alone adds robustness to single-joint noise.")

heading("3. Accuracy Results")
para(f"Overall accuracy: {acc*100:.2f}%  ({(res['ground_truth']==res['pred']).sum()}/{len(res)} "
     f"frames) against a manually-labeled per-frame ground truth.")
pdf.set_font("Courier", "", 9)
for line in ["Confusion matrix (rows=true, cols=pred):",
             "            sitting  standing",
             f"  sitting   {cm[0,0]:>7d}  {cm[0,1]:>8d}",
             f"  standing  {cm[1,0]:>7d}  {cm[1,1]:>8d}",
             f"Mismatched frames: {mism}"]:
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 5, line, **NEXT)
pdf.set_font("Helvetica", "", 10)

heading("4. Observations")
para("- MediaPipe detected the subject in all 613 frames; Savitzky-Golay smoothing removed jitter "
     "without distorting real motion.\n"
     "- Knee and hip angles separate the two activities cleanly; the elbow angle is comparatively "
     "uninformative for stand/sit.\n"
     "- All classification errors fall on transition frames (mid stand-up / sit-down), where the "
     "posture is genuinely ambiguous - the expected place for a hard threshold to disagree with a "
     "discrete human label.")

# ---- screenshots page ----
pdf.add_page()
heading("Demo Screenshots")
overlays = sorted([f for f in os.listdir(OUT) if f.startswith("overlay_")])
if len(overlays) >= 2:
    iw = 50
    ih = iw * 850 / 478
    y0 = pdf.get_y()
    pdf.image(os.path.join(OUT, overlays[0]), x=25, y=y0, w=iw)
    pdf.image(os.path.join(OUT, overlays[-1]), x=120, y=y0, w=iw)
    pdf.set_y(y0 + ih + 2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 5, "Skeleton overlay: standing (left) and sitting (right)", **NEXT)

for img, cap in [("smoothing.png", "Raw vs smoothed keypoint trajectory"),
                 ("angles.png", "Joint angles over time (dashed = transitions)"),
                 ("timeline.png", "Predicted vs ground truth")]:
    p = os.path.join(OUT, img)
    if os.path.exists(p):
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, cap, **NEXT)
        pdf.image(p, w=180)

pdf.output("report.pdf")
print(f"wrote report.pdf  (accuracy {acc*100:.2f}%)")
