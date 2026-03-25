import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# CONFIG
# -----------------------------
PRED_JSON = r"D:\DMS\Driver-Monitoring-System-main\runs\detect\val5\predictions.json"
GT_FOLDER = r"D:\DMS\Driver-Monitoring-System-main\dataset\valid\labels"
CLASS_ID_PHONE = 1

# -----------------------------
# LOAD PREDICTIONS
# -----------------------------
with open(PRED_JSON, "r") as f:
    preds = json.load(f)

# Extract phone predictions
pred_confs = []
for item in preds:  # Each image
    for det in item["boxes"]:  # YOLOv8 predictions boxes
        if det["class"] == CLASS_ID_PHONE:
            pred_confs.append(det["confidence"])

pred_confs = np.array(pred_confs)

# -----------------------------
# LOAD GROUND TRUTH (PHONE ONLY)
# -----------------------------
gt_files = [f for f in os.listdir(GT_FOLDER) if f.endswith(".txt")]
n_gt_phone = 0
for f in gt_files:
    with open(os.path.join(GT_FOLDER, f), "r") as file:
        lines = file.readlines()
        for line in lines:
            cid = int(line.split()[0])
            if cid == CLASS_ID_PHONE:
                n_gt_phone += 1

# -----------------------------
# COMPUTE METRICS VS CONFIDENCE
# -----------------------------
CONF_THRESHOLDS = np.linspace(0, 1, 50)
precisions, recalls, f1_scores = [], [], []

y_true = np.ones(n_gt_phone)
for t in CONF_THRESHOLDS:
    y_pred = (pred_confs >= t).astype(int)
    tp = min(sum(y_pred), n_gt_phone)  # TP cannot exceed number of GT
    fp = max(sum(y_pred) - tp, 0)
    fn = max(n_gt_phone - tp, 0)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(CONF_THRESHOLDS, precisions, label="Precision")
plt.plot(CONF_THRESHOLDS, recalls, label="Recall")
plt.plot(CONF_THRESHOLDS, f1_scores, label="F1-score", linestyle="--")
plt.xlabel("Confidence Threshold")
plt.ylabel("Metric Value")
plt.title("Phone Detection: Precision, Recall, F1 vs Confidence")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# CONFUSION MATRIX (at 0.5)
# -----------------------------
y_pred = (pred_confs >= 0.5).astype(int)
tp = min(sum(y_pred), n_gt_phone)
fp = max(sum(y_pred) - tp, 0)
fn = max(n_gt_phone - tp, 0)
tn = 0  # No explicit background counted

cm = np.array([[tp, fp],
               [fn, tn]])
disp = ConfusionMatrixDisplay(cm, display_labels=["phone", "background"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Phone Detection)")
plt.show()

# -----------------------------
# mAP BAR PLOT
# -----------------------------
# Use the numbers you calculated manually or update with auto-calculation

plt.figure(figsize=(5,5))
plt.bar(["phone"], [mAP50], width=0.4, label="mAP@0.5")
plt.bar(["phone"], [mAP5095], width=0.4, label="mAP@0.5:0.95", alpha=0.7)
plt.ylabel("mAP")
plt.title("Phone Detection mAP")
plt.legend()
plt.show()
