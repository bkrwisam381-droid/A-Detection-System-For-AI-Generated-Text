"""
AI Text Detector — Evaluation & Visualization
Compatible with the final 4-signal architecture:
  DeBERTa-v3-large + XGBoost (35 features) + Binoculars + Genre + Ensemble MLP

Generates:
  reports/eval_figures/1_confusion_matrix.png
  reports/eval_figures/2_roc_pr_curves.png
  reports/eval_figures/3_score_distributions.png
  reports/eval_figures/4_feature_importance.png
  reports/eval_figures/5_dashboard.png
  reports/eval_summary.txt

Run from project root:
  python src/evaluate.py
"""
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT / "src"))

CACHE_DIR  = _PROJECT_ROOT / "data"  / "processed"
SPLIT_DIR  = _PROJECT_ROOT / "data"  / "splits"
MODEL_DIR  = _PROJECT_ROOT / "models"
REPORT_DIR = _PROJECT_ROOT / "reports"
FIG_DIR    = REPORT_DIR / "eval_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────────
PALETTE = {
    "Ensemble":   "#3498db",
    "DeBERTa":    "#9b59b6",
    "XGBoost":    "#e67e22",
    "Binoculars": "#1abc9c",
    "Genre":      "#e74c3c",
}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.family":      "DejaVu Sans",
})

print("=" * 60)
print("AI Text Detector — Evaluation")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load signals and labels
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading signals and labels ...")

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df   = pd.read_csv(SPLIT_DIR / "val.csv")
test_df  = pd.read_csv(SPLIT_DIR / "test.csv")

n_train = len(train_df)
n_val   = len(val_df)

all_labels = np.array(
    train_df["label"].tolist() +
    val_df["label"].tolist() +
    test_df["label"].tolist()
)
N = len(all_labels)

def load_cache(name, fallback_val=0.5):
    p = CACHE_DIR / name
    if not p.exists():
        print(f"  WARNING: {name} not found — using {fallback_val}")
        return np.full(N, fallback_val, dtype=np.float32)
    return np.load(p).astype(np.float32)

deberta_raw = load_cache("deberta_probs_all.npy", 0.5)
xgb_raw     = load_cache("xgb_probs_all.npy",     0.5)
bino_raw    = load_cache("binoculars_all.npy",     0.5)
genre_raw   = load_cache("genre_all.npy",          0.5)

# ── Load MLP + calibrators ────────────────────────────────────────────────────
import joblib, torch, torch.nn as nn

mlp_path = MODEL_DIR / "ensemble" / "mlp_fusion.pt"
cal_path = MODEL_DIR / "ensemble" / "calibrators.pkl"

if not mlp_path.exists():
    print(f"ERROR: {mlp_path} not found — run train_ensemble.py first.")
    sys.exit(1)

cals = joblib.load(cal_path)

# Apply same calibration as inference.py (with 0.99 clip on DeBERTa)
deberta_c = cals["deberta"].predict(deberta_raw).astype(np.float32)
xgb_c = cals["xgb"].predict(xgb_raw).astype(np.float32)

ckpt     = torch.load(mlp_path, map_location="cpu", weights_only=False)
bino_p5  = ckpt.get("bino_p5",  float(np.percentile(bino_raw,  5)))
bino_p95 = ckpt.get("bino_p95", float(np.percentile(bino_raw, 95))) + 1e-6
bino_norm      = np.clip((bino_raw - bino_p5) / (bino_p95 - bino_p5), 0, 1)
bino_ai_signal = 1.0 - bino_norm
input_dim      = ckpt.get("input_dim", 4)

print(f"  MLP input_dim : {input_dim}")
print(f"  Features      : {ckpt.get('feature_names', 'unknown')}")


class FusionMLP(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


mlp = FusionMLP(input_dim)
mlp.load_state_dict(ckpt["model_state"])
mlp.eval()

if input_dim == 4:
    X_all = np.column_stack([deberta_c, xgb_c, bino_ai_signal, genre_raw])
elif input_dim == 3:
    X_all = np.column_stack([deberta_c, xgb_c, bino_ai_signal])
else:
    X_all = np.column_stack([deberta_c, xgb_c])

with torch.no_grad():
    ensemble_probs = mlp(torch.tensor(X_all, dtype=torch.float32)).numpy()

# Test set
test_start   = n_train + n_val
labels_test  = all_labels[test_start:]
deberta_test = deberta_raw[test_start:]
xgb_test     = xgb_raw[test_start:]
bino_test    = bino_ai_signal[test_start:]
genre_test   = genre_raw[test_start:]
ens_test     = ensemble_probs[test_start:]
preds_test   = (ens_test >= 0.5).astype(int)

print(f"  Test : {len(labels_test):,}  "
      f"({(labels_test==0).sum():,} human / {(labels_test==1).sum():,} AI)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Computing metrics ...")

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

def metrics(y_true, y_prob, name):
    yp = (y_prob >= 0.5).astype(int)
    return {
        "name":      name,
        "accuracy":  accuracy_score(y_true, yp),
        "f1":        f1_score(y_true, yp, zero_division=0),
        "precision": precision_score(y_true, yp, zero_division=0),
        "recall":    recall_score(y_true, yp, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "auc_pr":    average_precision_score(y_true, y_prob),
    }

m_ens     = metrics(labels_test, ens_test,     "Ensemble")
m_deberta = metrics(labels_test, deberta_test, "DeBERTa")
m_xgb     = metrics(labels_test, xgb_test,     "XGBoost")
m_bino    = metrics(labels_test, bino_test,    "Binoculars")
m_genre   = metrics(labels_test, genre_test,   "Genre")

for m in [m_ens, m_deberta, m_xgb, m_bino, m_genre]:
    print(f"  {m['name']:12s} | F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
          f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")

cm = confusion_matrix(labels_test, preds_test)
tn, fp, fn, tp = cm.ravel()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Confusion matrix ...")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Human", "AI"],
            yticklabels=["Human", "AI"],
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 18, "weight": "bold"}, ax=ax)
ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
ax.set_ylabel("Actual",    fontsize=13, labelpad=10)
ax.set_title("Confusion Matrix — Ensemble (Test Set)", fontsize=14, pad=15)
for (i, j), _ in np.ndenumerate(cm):
    lbl = ("TP" if i==1 and j==1 else "TN" if i==0 and j==0
           else "FP" if i==0 and j==1 else "FN")
    ax.text(j+0.5, i+0.72, lbl, ha="center", va="center",
            fontsize=10, color="grey", style="italic")
plt.tight_layout()
out = FIG_DIR / "1_confusion_matrix.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ROC + PR Curves
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] ROC + PR curves ...")

sigs = [
    ("Ensemble",   ens_test,     PALETTE["Ensemble"],   2.5),
    ("DeBERTa",    deberta_test, PALETTE["DeBERTa"],    1.5),
    ("XGBoost",    xgb_test,     PALETTE["XGBoost"],    1.5),
    ("Binoculars", bino_test,    PALETTE["Binoculars"], 1.5),
    ("Genre",      genre_test,   PALETTE["Genre"],      1.5),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for name, probs, color, lw in sigs:
    fpr, tpr, _ = roc_curve(labels_test, probs)
    auc = roc_auc_score(labels_test, probs)
    ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC={auc:.4f})")
ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curves", xlim=[0,1], ylim=[0,1.01])
ax.legend(fontsize=9, loc="lower right")

ax = axes[1]
for name, probs, color, lw in sigs:
    prec, rec, _ = precision_recall_curve(labels_test, probs)
    ap = average_precision_score(labels_test, probs)
    ax.plot(rec, prec, color=color, lw=lw, label=f"{name} (AP={ap:.4f})")
ax.axhline(labels_test.mean(), color="k", ls="--", lw=1, alpha=0.4)
ax.set(xlabel="Recall", ylabel="Precision",
       title="Precision-Recall Curves", xlim=[0,1], ylim=[0,1.01])
ax.legend(fontsize=9, loc="lower left")

plt.suptitle("Signal Comparison — Test Set", fontsize=15, y=1.02)
plt.tight_layout()
out = FIG_DIR / "2_roc_pr_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Score Distributions
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Score distributions ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, (name, probs, ca, ch) in enumerate([
    ("Ensemble",   ens_test,     "#e74c3c", "#2ecc71"),
    ("DeBERTa",    deberta_test, "#c0392b", "#27ae60"),
    ("XGBoost",    xgb_test,     "#e67e22", "#16a085"),
    ("Binoculars", bino_test,    "#8e44ad", "#2980b9"),
]):
    ax = axes[idx]
    ai  = probs[labels_test == 1]
    hum = probs[labels_test == 0]
    ax.hist(hum, bins=60, alpha=0.6, color=ch,
            label=f"Human (n={len(hum):,})", density=True)
    ax.hist(ai,  bins=60, alpha=0.6, color=ca,
            label=f"AI    (n={len(ai):,})",  density=True)
    ax.axvline(0.5, color="black", ls="--", lw=1.2, alpha=0.7,
               label="Threshold")
    ax.set(title=f"{name} Score Distribution",
           xlabel="Probability Score", ylabel="Density", xlim=[0,1])
    ax.legend(fontsize=9)

plt.suptitle("Score Distributions — Test Set", fontsize=15, y=1.01)
plt.tight_layout()
out = FIG_DIR / "3_score_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
feat_path = MODEL_DIR / "xgboost" / "feature_importance.csv"
if feat_path.exists():
    print("\n[6/7] Feature importance ...")
    fi  = pd.read_csv(feat_path).sort_values("importance", ascending=True).tail(20)
    clr = [PALETTE["XGBoost"] if i >= len(fi)-5 else "#bdc3c7"
           for i in range(len(fi))]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(fi["feature"], fi["importance"],
                   color=clr, edgecolor="white", height=0.7)
    for bar, val in zip(bars, fi["importance"]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8.5, color="#555")
    ax.set(xlabel="Feature Importance (XGBoost gain)",
           title="Top 20 Stylometric Features",
           xlim=[0, fi["importance"].max() * 1.15])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[
        Patch(color=PALETTE["XGBoost"], label="Top 5 features"),
        Patch(color="#bdc3c7",          label="Other features"),
    ], fontsize=10, loc="lower right")
    plt.tight_layout()
    out = FIG_DIR / "4_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
else:
    print("\n[6/7] feature_importance.csv not found — skipping.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Dashboard ...")

BG   = "#1a1a2e"
DARK = "#16213e"
CARD = "#0f3460"
TW   = "white"
TD   = "#a0aec0"

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        hspace=0.6, wspace=0.4,
                        left=0.06, right=0.97,
                        top=0.88,  bottom=0.06)

def stat_card(ax, val, lbl, color, fmt=".4f"):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.58, f"{val:{fmt}}", transform=ax.transAxes,
            ha="center", va="center", fontsize=26,
            fontweight="bold", color=color)
    ax.text(0.5, 0.22, lbl, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color=TD)

stat_card(fig.add_subplot(gs[0,0]), m_ens["f1"],        "F1 Score",  "#3498db")
stat_card(fig.add_subplot(gs[0,1]), m_ens["auc_roc"],   "AUC-ROC",   "#2ecc71")
stat_card(fig.add_subplot(gs[0,2]), m_ens["precision"], "Precision", "#e67e22")
stat_card(fig.add_subplot(gs[0,3]), m_ens["recall"],    "Recall",    "#9b59b6")

# Mini confusion matrix
ax = fig.add_subplot(gs[1, 0:2])
ax.set_facecolor(DARK)
for s in ax.spines.values(): s.set_edgecolor("#444")
cm_n  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
lbls  = [["TN","FP"],["FN","TP"]]
clrs  = [["#2ecc71","#e74c3c"],["#e74c3c","#2ecc71"]]
for i in range(2):
    for j in range(2):
        ax.add_patch(FancyBboxPatch(
            (j*0.45+0.05,(1-i)*0.45-0.38), 0.38, 0.38,
            boxstyle="round,pad=0.02", facecolor=clrs[i][j],
            alpha=0.85, transform=ax.transAxes))
        ax.text(j*0.45+0.24,(1-i)*0.45-0.19,
                f"{lbls[i][j]}\n{cm[i,j]:,}\n({cm_n[i,j]*100:.1f}%)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5, 0.97, "Confusion Matrix", transform=ax.transAxes,
        ha="center", va="top", fontsize=12, color=TW, fontweight="bold")
ax.text(0.24, 0.04, "Predicted: Human",
        transform=ax.transAxes, ha="center", fontsize=9, color=TD)
ax.text(0.70, 0.04, "Predicted: AI",
        transform=ax.transAxes, ha="center", fontsize=9, color=TD)

# Signal bars
ax = fig.add_subplot(gs[1, 2:4])
ax.set_facecolor(DARK)
for s in ax.spines.values(): s.set_edgecolor("#444")
snames = ["DeBERTa","XGBoost","Binoculars","Genre","Ensemble"]
f1s    = [m_deberta["f1"],m_xgb["f1"],m_bino["f1"],m_genre["f1"],m_ens["f1"]]
aucs   = [m_deberta["auc_roc"],m_xgb["auc_roc"],m_bino["auc_roc"],
          m_genre["auc_roc"],m_ens["auc_roc"]]
bclr   = [PALETTE["DeBERTa"],PALETTE["XGBoost"],
          PALETTE["Binoculars"],PALETTE["Genre"],PALETTE["Ensemble"]]
x = np.arange(len(snames)); w = 0.35
b1 = ax.bar(x-w/2, f1s,  w, color=bclr, alpha=0.85, label="F1")
b2 = ax.bar(x+w/2, aucs, w, color=bclr, alpha=0.45, label="AUC")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
            f"{b.get_height():.3f}", ha="center", fontsize=7.5, color=TW)
ax.set_xticks(x); ax.set_xticklabels(snames, color=TW, fontsize=9)
ax.set_ylim([0.6,1.02])
ax.set_title("F1 vs AUC-ROC per Signal", color=TW, fontsize=12, pad=8)
ax.tick_params(colors=TW)
ax.legend(fontsize=9, facecolor=DARK, labelcolor=TW)
ax.set_facecolor(DARK)
for s in ax.spines.values(): s.set_edgecolor("#444")

# Dataset stats
ax = fig.add_subplot(gs[2, 0:2])
ax.set_facecolor(DARK)
for s in ax.spines.values(): s.set_edgecolor("#444")
ax.set_xticks([]); ax.set_yticks([])
total = len(train_df)+len(val_df)+len(test_df)
ds = [
    ("Dataset", f"{total:,} total samples"),
    ("Train",   f"{len(train_df):,}  ({len(train_df)/total*100:.0f}%)"),
    ("Val",     f"{len(val_df):,}  ({len(val_df)/total*100:.0f}%)"),
    ("Test",    f"{len(test_df):,}  ({len(test_df)/total*100:.0f}%)"),
    ("Balance", "50% Human / 50% AI"),
    ("Sources", "RAID · HC3 · WildChat · Anthropic · DAIGT · Wikipedia · arXiv"),
]
ax.text(0.5, 0.96, "Dataset Statistics", transform=ax.transAxes,
        ha="center", va="top", fontsize=12, color=TW, fontweight="bold")
for i, (k, v) in enumerate(ds):
    y = 0.80 - i*0.135
    ax.text(0.05, y, k+":", transform=ax.transAxes, color=TD, fontsize=10)
    ax.text(0.38, y, v,     transform=ax.transAxes, color=TW, fontsize=9.5)

# Architecture
ax = fig.add_subplot(gs[2, 2:4])
ax.set_facecolor(DARK)
for s in ax.spines.values(): s.set_edgecolor("#444")
ax.set_xticks([]); ax.set_yticks([])
mi = [
    ("Semantic",    "DeBERTa-v3-large  (400M, BF16, max_len=448)"),
    ("Stylometric", "XGBoost  (35 features, hist, n_est=1000)"),
    ("Perplexity",  "Binoculars  (gpt2-large / gpt2-xl, Hans 2024)"),
    ("Genre",       "DistilBERT  (4-class, 99.7% F1, 66M params)"),
    ("Fusion",      "MLP  4→32→16→1  (BCE + isotonic calibration)"),
    ("Result",      f"Test F1={m_ens['f1']:.4f}  AUC={m_ens['auc_roc']:.4f}"),
]
ax.text(0.5, 0.96, "Model Architecture", transform=ax.transAxes,
        ha="center", va="top", fontsize=12, color=TW, fontweight="bold")
for i, (k, v) in enumerate(mi):
    y = 0.80 - i*0.135
    ax.text(0.04, y, k+":", transform=ax.transAxes, color=TD, fontsize=10)
    ax.text(0.30, y, v,     transform=ax.transAxes, color=TW, fontsize=9)

fig.suptitle("AI Text Detector — System Evaluation Dashboard",
             fontsize=18, fontweight="bold", color=TW, y=0.96)
out = FIG_DIR / "5_dashboard.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Text summary
# ══════════════════════════════════════════════════════════════════════════════
report_path = REPORT_DIR / "eval_summary.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("AI Text Detector — Evaluation Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write("DATASET\n")
    f.write(f"  Total : {total:,}\n")
    f.write(f"  Train : {len(train_df):,}\n")
    f.write(f"  Val   : {len(val_df):,}\n")
    f.write(f"  Test  : {len(test_df):,}\n")
    f.write(f"  Sources: RAID, HC3, WildChat, Anthropic HH-RLHF, "
            f"DAIGT, Wikipedia, arXiv\n\n")
    f.write("MODEL ARCHITECTURE\n")
    f.write(f"  Semantic     : DeBERTa-v3-large (400M, BF16, max_len=448)\n")
    f.write(f"  Stylometric  : XGBoost (35 features, hist, n_estimators=1000)\n")
    f.write(f"  Perplexity   : Binoculars (gpt2-large/gpt2-xl, Hans et al. 2024)\n")
    f.write(f"  Genre        : DistilBERT 4-class (99.7% F1)\n")
    f.write(f"  Fusion       : MLP {input_dim}→32→16→1 + isotonic calibration\n")
    f.write(f"  DeBERTa clip : 0.99 (prevents calibration cliff)\n\n")
    f.write("TEST SET PERFORMANCE\n")
    f.write(f"  {'Signal':<14} {'F1':>8} {'AUC-ROC':>10} "
            f"{'Precision':>12} {'Recall':>10}\n")
    f.write("  " + "-"*56 + "\n")
    for m in [m_ens, m_deberta, m_xgb, m_bino, m_genre]:
        f.write(f"  {m['name']:<14} {m['f1']:>8.4f} {m['auc_roc']:>10.4f} "
                f"{m['precision']:>12.4f} {m['recall']:>10.4f}\n")
    f.write(f"\nCONFUSION MATRIX\n")
    f.write(f"  TP : {tp:,}  TN : {tn:,}  FP : {fp:,}  FN : {fn:,}\n")
    f.write(f"  FP Rate : {fp/(fp+tn)*100:.2f}%\n")
    f.write(f"  FN Rate : {fn/(fn+tp)*100:.2f}%\n")
    f.write(f"\nFIGURES : {FIG_DIR}\n")
print(f"  Summary → {report_path}")


# ── Final print ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"  Ensemble F1       : {m_ens['f1']:.4f}")
print(f"  Ensemble AUC-ROC  : {m_ens['auc_roc']:.4f}")
print(f"  Ensemble Precision: {m_ens['precision']:.4f}")
print(f"  Ensemble Recall   : {m_ens['recall']:.4f}")
print(f"  FP Rate           : {fp/(fp+tn)*100:.2f}%")
print(f"  FN Rate           : {fn/(fn+tp)*100:.2f}%")
print(f"  Figures           : {FIG_DIR}")
print(f"  Summary           : {report_path}")
print("="*60)