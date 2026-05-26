"""Build a polished report.pdf for the pose activity classifier.

Regenerates publication-quality figures from results.csv, then composes a
designed multi-page report with reportlab (Platypus): title banner, stat cards,
a pipeline diagram, a color-coded confusion matrix, per-class metrics, a
decision-boundary scatter, and framed demo screenshots.
"""
import os
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)

OUT = "outputs"
ASSET = os.path.join(OUT, "report")
os.makedirs(ASSET, exist_ok=True)

# ----------------------------------------------------------------------------- palette
NAVY   = colors.HexColor("#10243E")
BLUE   = colors.HexColor("#1F6FB2")
TEAL   = colors.HexColor("#13A089")
GREEN  = colors.HexColor("#2E8B57")
RED    = colors.HexColor("#C0392B")
AMBER  = colors.HexColor("#E08A1E")
INK    = colors.HexColor("#1B2733")
GREY   = colors.HexColor("#5B6770")
LIGHT  = colors.HexColor("#EEF3F8")
LINE   = colors.HexColor("#D4DEE8")
WHITE  = colors.white
MPL = {"knee": "#1F6FB2", "hip": "#E08A1E", "elbow": "#13A089",
       "sit": "#C0392B", "stand": "#2E8B57"}

# ----------------------------------------------------------------------------- data
res = pd.read_csv("results.csv")
y_true, y_pred = res["ground_truth"], res["pred"]
acc = accuracy_score(y_true, y_pred)
labels = ["sitting", "standing"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
mism = res["frame"][y_true.values != y_pred.values].tolist()
N = len(res)
correct = int((y_true == y_pred).sum())

meta = {}
for ln in open(os.path.join(OUT, "metrics.txt")):
    if "=" in ln:
        k, v = ln.strip().split("=", 1)
        meta[k] = v
transitions = ast.literal_eval(meta.get("transitions", "[]"))
KNEE_T = float(meta.get("knee_thresh", 130))
HIP_T = float(meta.get("hip_thresh", 130))


def intervals(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


stand_iv = intervals((y_true == "standing").values)

# ----------------------------------------------------------------------------- figures
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#9AA7B2", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E3E9EF", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def shade(ax):
    for a, b in stand_iv:
        ax.axvspan(a, b, color=GREEN.rgb(), alpha=0.08, lw=0)


# --- angles over time ---
fa = (10, 2.8)
fig, ax = plt.subplots(figsize=fa)
shade(ax)
for c in ["knee", "hip", "elbow"]:
    ax.plot(res["frame"], res[c], color=MPL[c], lw=1.6, label=f"{c} angle")
ax.axhline(KNEE_T, color=GREY.rgb(), ls=":", lw=1, alpha=0.8)
for t in transitions:
    ax.axvline(t, color="#3A4A5A", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("frame"); ax.set_ylabel("angle (deg)")
ax.set_xlim(0, N - 1); ax.set_ylim(55, 190)
ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=9)
ax.set_title("Joint angles over time  -  green = standing (ground truth), dashed = transitions",
             fontsize=10, color="#1B2733", pad=8)
fig.tight_layout(); fig.savefig(os.path.join(ASSET, "angles.png"), bbox_inches="tight"); plt.close(fig)

# --- timeline ---
ft = (10, 2.2)
fig, ax = plt.subplots(figsize=ft)
shade(ax)
num = {"sitting": 0, "standing": 1}
gt = y_true.map(num); pr = y_pred.map(num)
ax.step(res["frame"], gt, where="post", color=BLUE.rgb(), lw=2, label="ground truth")
ax.step(res["frame"], pr + 0.03, where="post", color=AMBER.rgb(), lw=1.8, label="predicted", alpha=0.9)
ax.scatter(mism, [0.5] * len(mism), color=RED.rgb(), s=28, zorder=5, label="mismatch")
ax.set_yticks([0, 1]); ax.set_yticklabels(["sitting", "standing"])
ax.set_xlabel("frame"); ax.set_xlim(0, N - 1); ax.set_ylim(-0.2, 1.2)
ax.legend(loc="center right", frameon=False, fontsize=9)
ax.set_title(f"Predicted vs ground truth  -  accuracy {acc*100:.2f}%", fontsize=10, color="#1B2733", pad=8)
fig.tight_layout(); fig.savefig(os.path.join(ASSET, "timeline.png"), bbox_inches="tight"); plt.close(fig)

# --- decision boundary scatter (knee vs hip) ---
fd = (6.6, 3.2)
fig, ax = plt.subplots(figsize=fd)
for lab, col in [("sitting", MPL["sit"]), ("standing", MPL["stand"])]:
    sub = res[res["ground_truth"] == lab]
    ax.scatter(sub["knee"], sub["hip"], s=14, alpha=0.55, color=col, label=lab, edgecolors="none")
ax.axvline(KNEE_T, color=GREY.rgb(), ls="--", lw=1.2)
ax.axhline(HIP_T, color=GREY.rgb(), ls="--", lw=1.2)
ax.axvspan(KNEE_T, 190, ymin=(HIP_T - 60) / (190 - 60), color=GREEN.rgb(), alpha=0.06, lw=0)
ax.text(KNEE_T + 3, 64, f"knee = {KNEE_T:.0f}deg", color=GREY.rgb(), fontsize=8)
ax.text(62, HIP_T + 2, f"hip = {HIP_T:.0f}deg", color=GREY.rgb(), fontsize=8)
ax.text(176, 184, "STANDING\nregion", ha="right", va="top", color=GREEN.rgb(), fontsize=9, weight="bold")
ax.set_xlabel("knee angle (deg)"); ax.set_ylabel("hip angle (deg)")
ax.set_xlim(60, 190); ax.set_ylim(60, 190)
ax.legend(loc="lower left", frameon=False, fontsize=9, title="ground truth")
ax.set_title("Why the rule works: angle separation by activity", fontsize=10, color="#1B2733", pad=8)
fig.tight_layout(); fig.savefig(os.path.join(ASSET, "scatter.png"), bbox_inches="tight"); plt.close(fig)

# --- confusion matrix heatmap ---
fc = (3.6, 3.2)
fig, ax = plt.subplots(figsize=fc)
ax.imshow([[1, 0], [0, 1]], cmap=matplotlib.colors.ListedColormap(["#F6DAD6", "#D6ECDD"]),
          vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        good = (i == j)
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=20, weight="bold", color=(GREEN.rgb() if good else RED.rgb()))
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["sitting", "standing"]); ax.set_yticklabels(["sitting", "standing"])
ax.set_xlabel("predicted"); ax.set_ylabel("actual")
ax.set_title("Confusion matrix", fontsize=10, color="#1B2733", pad=8)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-.5, 2, 1), minor=True); ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", lw=3); ax.tick_params(which="minor", length=0)
fig.tight_layout(); fig.savefig(os.path.join(ASSET, "cm.png"), bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- styles
styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, textColor=WHITE, leading=23),
    "sub":   ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#C7D6E5"), leading=13),
    "h":     ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=12.5, textColor=NAVY, leading=15),
    "body":  ParagraphStyle("b", fontName="Helvetica", fontSize=9.6, textColor=INK, leading=14, alignment=4, spaceAfter=5),
    "cap":   ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=8.5, textColor=GREY, leading=11, spaceBefore=3, spaceAfter=2),
    "cardN": ParagraphStyle("cn", fontName="Helvetica-Bold", fontSize=19, alignment=1, leading=21),
    "cardL": ParagraphStyle("cl", fontName="Helvetica", fontSize=7.6, textColor=GREY, alignment=1, leading=9),
    "stage": ParagraphStyle("st", fontName="Helvetica-Bold", fontSize=8.2, textColor=NAVY, alignment=1, leading=10),
    "arrow": ParagraphStyle("ar", fontName="Helvetica-Bold", fontSize=14, textColor=BLUE, alignment=1),
    "th":    ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.6, textColor=WHITE, alignment=1, leading=11),
    "td":    ParagraphStyle("td", fontName="Helvetica", fontSize=8.8, textColor=INK, alignment=1, leading=11),
    "tdl":   ParagraphStyle("tdl", fontName="Helvetica-Bold", fontSize=8.8, textColor=INK, leading=11),
}

doc = SimpleDocTemplate("report.pdf", pagesize=A4,
                        leftMargin=16 * mm, rightMargin=16 * mm,
                        topMargin=14 * mm, bottomMargin=16 * mm,
                        title="Pose-Based Human Activity Classifier",
                        author="CLO-3 Computer Vision")
EPW = doc.width


def banner():
    head = Table([[Paragraph("Pose-Based Human Activity Classifier", S["title"])],
                  [Paragraph("Computer Vision &nbsp;|&nbsp; Complex Computing Problem (CLO-3) &nbsp;|&nbsp; "
                             "Rule-based pose pipeline with MediaPipe", S["sub"])]],
                 colWidths=[EPW])
    head.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LINEBELOW", (0, 0), (-1, -1), 3, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (0, 0), 12), ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
    ]))
    return head


def stat_cards():
    data = [("99.35%" if False else f"{acc*100:.2f}%", "Accuracy", GREEN),
            (str(N), "Frames analysed", BLUE),
            ("2", "Activities", TEAL),
            (f"{len(mism)}", "Misclassified", RED)]
    cells = []
    for val, lab, col in data:
        c = Table([[Paragraph(f'<font color="#{col.hexval()[2:]}">{val}</font>', S["cardN"])],
                   [Paragraph(lab, S["cardL"])]], colWidths=[(EPW - 18) / 4])
        c.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("LINEABOVE", (0, 0), (-1, 0), 3, col),
            ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
            ("TOPPADDING", (0, 1), (-1, 1), 0), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ]))
        cells.append(c)
    row = Table([cells], colWidths=[(EPW) / 4] * 4)
    row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 3),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                             ("TOPPADDING", (0, 0), (-1, -1), 0),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return row


def section(num, title):
    badge = Table([[Paragraph(f'<font color="white"><b>{num}</b></font>', S["cardN"])]],
                  colWidths=[8 * mm], rowHeights=[8 * mm])
    badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    badge_num = Paragraph(str(num), ParagraphStyle("bn", fontName="Helvetica-Bold",
                          fontSize=12, textColor=WHITE, alignment=1))
    badge = Table([[badge_num]], colWidths=[8 * mm], rowHeights=[8 * mm])
    badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    t = Table([[badge, Paragraph(title, S["h"])]], colWidths=[8 * mm, EPW - 8 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (1, 0), (1, 0), 8),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.8, LINE),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    return t


def pipeline():
    stages = ["Pose\nDetection", "Savitzky-Golay\nSmoothing",
              "Joint Angle\nComputation", "Rule-Based\nClassification"]
    cols = [Paragraph(s.replace("\n", "<br/>"), S["stage"]) for s in stages]
    row, widths = [], []
    sw = (EPW - 3 * 9 * mm) / 4
    for i, c in enumerate(cols):
        row.append(c); widths.append(sw)
        if i < 3:
            row.append(Paragraph("&rarr;", S["arrow"])); widths.append(9 * mm)
    t = Table([row], colWidths=widths, rowHeights=[14 * mm])
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i in range(4):
        col = i * 2
        sty += [("BACKGROUND", (col, 0), (col, 0), LIGHT),
                ("LINEABOVE", (col, 0), (col, 0), 2.5, [BLUE, TEAL, AMBER, GREEN][i]),
                ("BOX", (col, 0), (col, 0), 0.5, LINE)]
    t.setStyle(TableStyle(sty))
    return t


def metrics_table(avail_w=None):
    if avail_w is None:
        avail_w = EPW
    head = [Paragraph(h, S["th"]) for h in ["Class", "Precision", "Recall", "F1-score", "Support"]]
    rows = [head]
    for lab in labels:
        r = rep[lab]
        rows.append([Paragraph(lab.capitalize(), S["tdl"]),
                     Paragraph(f"{r['precision']:.3f}", S["td"]),
                     Paragraph(f"{r['recall']:.3f}", S["td"]),
                     Paragraph(f"{r['f1-score']:.3f}", S["td"]),
                     Paragraph(f"{int(r['support'])}", S["td"])])
    ma = rep["macro avg"]
    rows.append([Paragraph("Macro avg", S["tdl"]),
                 Paragraph(f"{ma['precision']:.3f}", S["td"]),
                 Paragraph(f"{ma['recall']:.3f}", S["td"]),
                 Paragraph(f"{ma['f1-score']:.3f}", S["td"]),
                 Paragraph(f"{N}", S["td"])])
    w = [avail_w * 0.30] + [avail_w * 0.175] * 4
    t = Table(rows, colWidths=w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, LINE),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def framed(path, w, h):
    img = Image(path, width=w, height=h)
    t = Table([[img]], colWidths=[w])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.7, LINE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def fig_img(name, fw, fh):
    return Image(os.path.join(ASSET, name), width=EPW, height=EPW * fh / fw)


# ----------------------------------------------------------------------------- story
story = [banner(), Spacer(1, 8), stat_cards(), Spacer(1, 12)]

story.append(section(1, "Overview"))
story.append(Paragraph(
    "This project detects human poses from a video, analyses joint geometry, and classifies the "
    "activity in every frame. <b>Video source:</b> own pre-recorded clip <b>activity.mp4</b> "
    "(~20.5&nbsp;s, 613 frames @ 30&nbsp;fps) of a single person repeatedly standing up and "
    "sitting down. The four-stage pipeline is shown below.", S["body"]))
story.append(Spacer(1, 4))
story.append(pipeline())
story.append(Spacer(1, 12))

story.append(section(2, "Methodology"))
story.append(Paragraph(
    "<b>Pose detection &amp; pre-processing.</b> The pre-trained MediaPipe Pose model returns 33 "
    "landmarks per frame. Missing detections are linearly interpolated and each landmark trajectory "
    "is de-jittered with a Savitzky&ndash;Golay filter (window 11, order 3) along the time axis, "
    "which acts as a low-pass smoother that preserves real motion.", S["body"]))
story.append(Paragraph(
    "<b>Joint-angle computation.</b> For three landmarks A&ndash;B&ndash;C, the angle at B is "
    "&theta; = arccos((BA&middot;BC)/(|BA||BC|)), evaluated in pixel coordinates. Three angles are "
    "tracked (left/right averaged): <b>knee</b> (hip&ndash;knee&ndash;ankle), <b>hip</b> "
    "(shoulder&ndash;hip&ndash;knee) and <b>elbow</b> (shoulder&ndash;elbow&ndash;wrist).", S["body"]))
story.append(Paragraph(
    f"<b>Rule-based classifier.</b> A frame is labelled <b>STANDING</b> when "
    f"(knee &gt; {KNEE_T:.0f}&deg;) AND (hip &gt; {HIP_T:.0f}&deg;), otherwise <b>SITTING</b>. "
    "Using two joints rather than the knee alone adds robustness to single-joint noise.", S["body"]))

story.append(section(3, "Results"))
story.append(Paragraph(
    f"The classifier reaches <b>{acc*100:.2f}%</b> accuracy ({correct}/{N} frames) against a "
    f"manually-labelled per-frame ground truth. The confusion matrix and per-class metrics below "
    f"show the only errors are {len(mism)} transition frames {mism}.", S["body"]))

_mt_w = EPW - 68 * mm - 8
cm_block = Table([[framed(os.path.join(ASSET, "cm.png"), 62 * mm, 55 * mm), metrics_table(_mt_w)]],
                 colWidths=[68 * mm, EPW - 68 * mm])
cm_block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (1, 0), (1, 0), 6)]))
story.append(cm_block)
story.append(Spacer(1, 10))

SW = EPW * 0.82
story.append(KeepTogether([Paragraph("Activity separation in knee&ndash;hip angle space.", S["cap"]),
                           Image(os.path.join(ASSET, "scatter.png"), width=SW, height=SW * 3.2 / 6.6)]))
story.append(Spacer(1, 6))
story.append(KeepTogether([Paragraph("Joint angles tracked across the clip.", S["cap"]),
                           fig_img("angles.png", 10, 3.6)]))

story.append(PageBreak())
story.append(section(4, "Tracking, Transitions & Demo"))
story.append(KeepTogether([Paragraph("Per-frame prediction vs ground truth.", S["cap"]),
                           fig_img("timeline.png", 10, 2.2)]))
story.append(Spacer(1, 4))
story.append(KeepTogether([Paragraph("Smoothing reduces keypoint jitter (right-knee vertical position).", S["cap"]),
                           framed(os.path.join(OUT, "smoothing.png"), EPW, EPW * 3.5 / 11)]))
story.append(Spacer(1, 8))

# skeleton overlays
overlays = sorted([f for f in os.listdir(OUT) if f.startswith("overlay_")])
if len(overlays) >= 2:
    ow = 46 * mm; oh = ow * 850 / 478
    pair = Table([[framed(os.path.join(OUT, overlays[0]), ow, oh),
                   framed(os.path.join(OUT, overlays[-1]), ow, oh)]],
                 colWidths=[EPW / 2, EPW / 2])
    pair.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(KeepTogether([
        Paragraph("Skeleton overlay produced by the pipeline: standing (left) and sitting (right).", S["cap"]),
        pair]))
    story.append(Spacer(1, 10))

story.append(section(5, "Observations"))
for o in [
    "MediaPipe detected the subject in all 613 frames; Savitzky&ndash;Golay smoothing removed "
    "jitter without distorting the underlying motion.",
    "Knee and hip angles separate the two activities cleanly (standing ~150&ndash;180&deg;, "
    "sitting ~65&ndash;120&deg;); the elbow angle is comparatively uninformative for stand/sit.",
    "Every classification error falls on a transition frame (mid stand-up / sit-down), where the "
    "posture is genuinely ambiguous &mdash; exactly where a hard threshold is expected to disagree "
    "with a discrete human label.",
]:
    story.append(Paragraph(f'<font color="#1F6FB2"><b>&#9632;</b></font>&nbsp; {o}', S["body"]))


# ----------------------------------------------------------------------------- footer
def footer(canvas, d):
    canvas.saveState()
    canvas.setStrokeColor(LINE); canvas.setLineWidth(0.6)
    y = 12 * mm
    canvas.line(d.leftMargin, y, d.leftMargin + d.width, y)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(d.leftMargin, y - 4 * mm,
                      "Pose-Based Human Activity Classifier  |  CLO-3 Computer Vision")
    canvas.drawRightString(d.leftMargin + d.width, y - 4 * mm, f"Page {d.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote report.pdf  (accuracy {acc*100:.2f}%, {doc.page} pages)")
