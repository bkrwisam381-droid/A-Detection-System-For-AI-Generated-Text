"""
AI Text Detector — Ensemble MLP Training
Signals:
  1. DeBERTa-v3-large probability  (calibrated)
  2. XGBoost stylometric probability (calibrated)
  3. Binoculars perplexity score    (Hans et al. 2024 — gpt2-large / gpt2-xl)

Why Binoculars instead of Qwen perplexity:
  Qwen2.5 is a strong modern model — GPT-2/3 outputs look "rough" to it
  and get HIGH perplexity (same as human text), killing the signal.
  Binoculars uses two GPT-2 models that share the WebText training
  distribution with virtually every AI generator. The observer/scorer
  RATIO cancels topic and length effects, making it robust across
  all generators in the dataset.
"""
import logging
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from features import FeatureExtractor

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR      = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "train_ensemble.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────────
class EnsembleConfig:
    train_csv        = _PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv          = _PROJECT_ROOT / "data" / "splits" / "val.csv"
    test_csv         = _PROJECT_ROOT / "data" / "splits" / "test.csv"
    deberta_dir      = _PROJECT_ROOT / "models" / "deberta" / "best_model"
    xgb_model_path   = _PROJECT_ROOT / "models" / "xgboost" / "xgb_model.json"
    xgb_scaler_path  = _PROJECT_ROOT / "models" / "xgboost" / "scaler.pkl"
    xgb_active_feats = _PROJECT_ROOT / "models" / "xgboost" / "active_features.json"
    output_dir       = _PROJECT_ROOT / "models" / "ensemble"
    signal_cache     = _PROJECT_ROOT / "data" / "processed"

    deberta_batch = 32

    # MLP
    mlp_epochs = 50
    mlp_lr     = 1e-3
    mlp_batch  = 256
    patience   = 10
    seed       = 42

cfg = EnsembleConfig()
cfg.output_dir.mkdir(parents=True, exist_ok=True)
cfg.signal_cache.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fusion MLP  (3 inputs: DeBERTa + XGBoost + Binoculars)
# ══════════════════════════════════════════════════════════════════════════════
class FusionMLP(nn.Module):
    """
    Input features (4):
        0 — DeBERTa AI probability   (IsotonicRegression calibrated)
        1 — XGBoost AI probability   (IsotonicRegression calibrated)
        2 — Binoculars AI signal     (normalised + inverted: high = AI)
        3 — Genre formality score    (P(encyclopedic) + P(academic))
              high = formal text where DeBERTa is unreliable
              low  = informal text where DeBERTa is trustworthy

    The genre signal teaches the MLP context-dependent signal weighting:
    when formality is high, discount DeBERTa and trust Binoculars more.

    Architecture: 4 → 32 → 16 → 1
    """
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Signal extractors
# ══════════════════════════════════════════════════════════════════════════════
def deberta_predict_proba(texts: list[str]) -> np.ndarray:
    """DeBERTa-v3-large AI probabilities. Cached after first run."""
    cache = cfg.signal_cache / "deberta_probs_all.npy"
    if cache.exists():
        log.info("Loading cached DeBERTa probabilities ...")
        return np.load(cache)

    log.info("Running DeBERTa inference on %d texts ...", len(texts))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.deberta_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(cfg.deberta_dir),
        dtype=torch.bfloat16,
    ).to(device).eval()

    from tqdm import tqdm
    all_probs = []
    for i in tqdm(range(0, len(texts), cfg.deberta_batch), desc="DeBERTa"):
        batch = texts[i : i + cfg.deberta_batch]
        enc   = tokenizer(batch, max_length=512, truncation=True,
                          padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(
                model(**enc).logits, dim=-1
            )[:, 1].float().cpu().numpy()
        all_probs.extend(probs.tolist())

    del model
    torch.cuda.empty_cache()

    result = np.array(all_probs, dtype=np.float32)
    np.save(cache, result)
    log.info("DeBERTa probs cached → %s", cache)
    return result


def xgb_predict_proba(texts: list[str]) -> np.ndarray:
    """XGBoost stylometric AI probabilities. Cached after first run."""
    cache = cfg.signal_cache / "xgb_probs_all.npy"
    if cache.exists():
        log.info("Loading cached XGBoost probabilities ...")
        return np.load(cache)

    log.info("Running XGBoost inference on %d texts ...", len(texts))
    model  = xgb.XGBClassifier()
    model.load_model(cfg.xgb_model_path)
    scaler = joblib.load(cfg.xgb_scaler_path)

    if cfg.xgb_active_feats.exists():
        with open(cfg.xgb_active_feats) as f:
            feature_cols = json.load(f)
        log.info("Using %d active XGB features", len(feature_cols))
    else:
        feature_cols = FeatureExtractor.FEATURE_NAMES

    extractor = FeatureExtractor()
    feat_df   = extractor.extract_dataframe(texts, desc="XGB features")
    X         = feat_df[feature_cols].values
    probs     = model.predict_proba(scaler.transform(X))[:, 1].astype(np.float32)

    np.save(cache, probs)
    log.info("XGBoost probs cached → %s", cache)
    return probs


def binoculars_scores(texts: list[str]) -> np.ndarray:
    """
    Binoculars score (Hans et al. 2024 — arxiv.org/abs/2401.12070).

    score = CE_loss(gpt2-large, text) / CE_loss(gpt2-xl, text)

    Low ratio  → text is more predictable to gpt2-xl than gpt2-large
               → AI-generated (xl overfits to AI writing patterns)
    High ratio → text surprises gpt2-xl more than gpt2-large → human

    Why this works across ALL generators including GPT-2/3:
      Both models share the WebText training distribution with every major
      AI generator. The ratio cancels topic/length/domain effects that
      make raw perplexity unreliable. Qwen2.5 failed because GPT-2/3 text
      looks rough to a modern LLM — the signal inverted for older generators.

    VRAM: gpt2-large (~800MB) + gpt2-xl (~6GB) in fp16 ≈ 6.8GB.
          Loaded after DeBERTa inference is done and VRAM is free.
    """
    cache = cfg.signal_cache / "binoculars_all.npy"
    if cache.exists():
        log.info("Loading cached Binoculars scores ...")
        arr = np.load(cache)
        log.info("  shape=%s  mean=%.3f  std=%.3f", arr.shape, arr.mean(), arr.std())
        return arr

    log.info("Computing Binoculars scores on %d texts ...", len(texts))
    log.info("Loading gpt2-large (observer) + gpt2-xl (scorer) ...")

    from transformers import AutoModelForCausalLM
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        log.info("Loading gpt2-large ...")
        observer = AutoModelForCausalLM.from_pretrained(
            "gpt2-large", torch_dtype=torch.float16,
        ).to(device).eval()

        log.info("Loading gpt2-xl ...")
        scorer = AutoModelForCausalLM.from_pretrained(
            "gpt2-xl", torch_dtype=torch.float16,
        ).to(device).eval()

        if torch.cuda.is_available():
            log.info("VRAM after loading: %.2f GB",
                     torch.cuda.memory_allocated() / 1e9)

    except Exception as e:
        log.warning("Could not load GPT-2 models: %s", e)
        log.warning("Returning neutral Binoculars scores (0.5)")
        neutral = np.full(len(texts), 0.5, dtype=np.float32)
        np.save(cache, neutral)
        return neutral

    scores = []
    for text in tqdm(texts, desc="Binoculars"):
        try:
            enc = tokenizer(
                text, max_length=512, truncation=True,
                return_tensors="pt", padding=False,
            ).to(device)

            if enc["input_ids"].shape[1] < 2:
                scores.append(1.0)
                continue

            with torch.no_grad():
                ce_obs   = observer(**enc, labels=enc["input_ids"]).loss.item()
                ce_score = scorer(**enc,   labels=enc["input_ids"]).loss.item()

            scores.append(ce_obs / max(ce_score, 1e-6))

        except Exception:
            scores.append(1.0)

    del observer, scorer
    torch.cuda.empty_cache()
    log.info("Freed GPT-2 models from VRAM.")

    arr = np.array(scores, dtype=np.float32)
    log.info("Binoculars — mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
             arr.mean(), arr.std(), arr.min(), arr.max())

    np.save(cache, arr)
    log.info("Binoculars scores cached → %s", cache)
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# Genre formality scoring
# ══════════════════════════════════════════════════════════════════════════════
def genre_scores(texts: list[str]) -> np.ndarray:
    """
    Compute formality_score for each text using the fine-tuned genre classifier.
    formality_score = P(encyclopedic) + P(academic)
    Cached after first run.
    """
    cache = cfg.signal_cache / "genre_all.npy"
    if cache.exists():
        log.info("Loading cached genre scores ...")
        arr = np.load(cache)
        log.info("  shape=%s  mean=%.3f  std=%.3f", arr.shape, arr.mean(), arr.std())
        return arr

    log.info("Computing genre scores on %d texts ...", len(texts))

    sys.path.append(str(Path(__file__).resolve().parent))
    from genre_classifier import GenreClassifier

    gc = GenreClassifier()
    gc.load()
    arr = gc.predict_batch(texts, batch_size=64)

    del gc.model
    torch.cuda.empty_cache()
    log.info("Freed genre model from VRAM.")

    log.info("Genre scores — mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
             arr.mean(), arr.std(), arr.min(), arr.max())
    np.save(cache, arr)
    log.info("Genre scores cached → %s", cache)
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_probabilities(
    deberta_probs:     np.ndarray,
    xgb_probs:         np.ndarray,
    val_probs_deberta: np.ndarray,
    val_probs_xgb:     np.ndarray,
    val_labels:        np.ndarray,
):
    """
    Calibrate raw probabilities with IsotonicRegression fitted on the val set.
    Returns calibrated arrays for all splits + the fitted calibrators
    (calibrators must be saved and reused at inference time).
    """
    log.info("Calibrating probabilities with isotonic regression ...")

    deberta_cal = IsotonicRegression(out_of_bounds="clip")
    deberta_cal.fit(val_probs_deberta, val_labels)
    deberta_calibrated = deberta_cal.predict(deberta_probs).astype(np.float32)

    xgb_cal = IsotonicRegression(out_of_bounds="clip")
    xgb_cal.fit(val_probs_xgb, val_labels)
    xgb_calibrated = xgb_cal.predict(xgb_probs).astype(np.float32)

    log.info("Calibration complete.")
    return deberta_calibrated, xgb_calibrated, deberta_cal, xgb_cal


# ══════════════════════════════════════════════════════════════════════════════
# Signal matrix
# ══════════════════════════════════════════════════════════════════════════════
def build_signal_matrix(
    deberta_probs:  np.ndarray,
    xgb_probs:      np.ndarray,
    bino_scores:    np.ndarray,
    genre_formality: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Assemble the 4-feature matrix for the MLP.
    Columns:
      0 - DeBERTa calibrated probability
      1 - XGBoost calibrated probability
      2 - Binoculars AI signal (normalised + inverted)
      3 - Genre formality score (P(encyclopedic) + P(academic))
    Returns (matrix, bino_p5, bino_p95).
    """
    p5  = float(np.percentile(bino_scores,  5))
    p95 = float(np.percentile(bino_scores, 95)) + 1e-6
    bino_norm      = np.clip((bino_scores - p5) / (p95 - p5), 0, 1)
    bino_ai_signal = 1.0 - bino_norm   # invert: low ratio → high AI signal

    X = np.column_stack([
        deberta_probs,
        xgb_probs,
        bino_ai_signal,
        genre_formality,   # high = encyclopedic/academic → DeBERTa unreliable
    ]).astype(np.float32)

    return X, p5, p95


# ══════════════════════════════════════════════════════════════════════════════
# MLP training
# ══════════════════════════════════════════════════════════════════════════════
def train_mlp(X_train, y_train, X_val, y_val) -> FusionMLP:
    torch.manual_seed(cfg.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = FusionMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.mlp_lr)
    criterion = nn.BCELoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=cfg.mlp_batch, shuffle=True,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_loss, patience_ctr, best_state = float("inf"), 0, None

    log.info("Training MLP fusion head (%d inputs) for up to %d epochs ...",
             X_train.shape[1], cfg.mlp_epochs)

    for epoch in range(cfg.mlp_epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss  = criterion(model(X_val_t), y_val_t).item()
            val_preds = (model(X_val_t).cpu().numpy() >= 0.5).astype(int)
            val_f1    = f1_score(y_val, val_preds, zero_division=0)

        if (epoch + 1) % 10 == 0:
            log.info("Epoch %3d | val_loss=%.4f | val_f1=%.4f",
                     epoch + 1, val_loss, val_f1)

        if val_loss < best_loss:
            best_loss    = val_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    return model


def eval_mlp(model: FusionMLP, X: np.ndarray, y: np.ndarray, name: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    m = {
        "split":     name,
        "accuracy":  accuracy_score(y, preds),
        "f1":        f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "auc_roc":   roc_auc_score(y, probs),
    }
    log.info("%s | acc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f  auc=%.4f",
             name.upper().ljust(6),
             m["accuracy"], m["f1"], m["precision"], m["recall"], m["auc_roc"])
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run():
    log.info("=" * 60)
    log.info("Ensemble MLP Fusion Training  (DeBERTa + XGBoost + Binoculars)")
    log.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if not cfg.deberta_dir.exists():
        log.error("DeBERTa model not found — run train_deberta.py first.")
        return
    if not cfg.xgb_model_path.exists():
        log.error("XGBoost model not found — run train_xgboost.py first.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading data ...")
    train_df = pd.read_csv(cfg.train_csv)
    val_df   = pd.read_csv(cfg.val_csv)
    test_df  = pd.read_csv(cfg.test_csv)

    all_texts  = (train_df["text"].tolist() +
                  val_df["text"].tolist() +
                  test_df["text"].tolist())
    all_labels = (train_df["label"].tolist() +
                  val_df["label"].tolist() +
                  test_df["label"].tolist())
    n_train, n_val = len(train_df), len(val_df)
    log.info("Total texts: %d  (train=%d  val=%d  test=%d)",
             len(all_texts), n_train, n_val, len(test_df))

    # ── Extract signals ───────────────────────────────────────────────────────
    log.info(" ")
    log.info("Step 1/4 — DeBERTa inference ...")
    deberta_probs = deberta_predict_proba(all_texts)

    log.info(" ")
    log.info("Step 2/4 — XGBoost inference ...")
    xgb_probs = xgb_predict_proba(all_texts)

    log.info(" ")
    log.info("Step 3/5 — Binoculars perplexity scoring ...")
    bino = binoculars_scores(all_texts)

    log.info(" ")
    log.info("Step 4/5 — Genre classification ...")
    log.info("(DistilBERT — encyclopedic/academic/informal/general)")
    genre = genre_scores(all_texts)

    # ── Calibrate ─────────────────────────────────────────────────────────────
    log.info(" ")
    log.info("Step 5/5 — Calibrating probabilities ...")

    val_start  = n_train
    val_end    = n_train + n_val
    val_labels = np.array(all_labels[val_start:val_end])

    deberta_cal_arr, xgb_cal_arr, deberta_cal, xgb_cal = calibrate_probabilities(
        deberta_probs, xgb_probs,
        deberta_probs[val_start:val_end],
        xgb_probs[val_start:val_end],
        val_labels,
    )

    # ── Build signal matrix ───────────────────────────────────────────────────
    log.info(" ")
    log.info("Building signal matrix ...")
    X_all, bino_p5, bino_p95 = build_signal_matrix(
        deberta_cal_arr, xgb_cal_arr, bino, genre
    )
    y_all = np.array(all_labels, dtype=np.float32)

    X_train = X_all[:n_train];               y_train = y_all[:n_train]
    X_val   = X_all[n_train:n_train+n_val];  y_val   = y_all[n_train:n_train+n_val]
    X_test  = X_all[n_train+n_val:];         y_test  = y_all[n_train+n_val:]

    log.info("Signal matrix: %s  (DeBERTa_cal | XGB_cal | Binoculars_ai | Genre_formality)",
             X_all.shape)

    # ── Train MLP ─────────────────────────────────────────────────────────────
    log.info(" ")
    mlp = train_mlp(X_train, y_train, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info(" ")
    log.info("-- Final evaluation --")
    train_m = eval_mlp(mlp, X_train, y_train, "train")
    val_m   = eval_mlp(mlp, X_val,   y_val,   "val")
    test_m  = eval_mlp(mlp, X_test,  y_test,  "test")

    # ── Save MLP ──────────────────────────────────────────────────────────────
    mlp_path = cfg.output_dir / "mlp_fusion.pt"
    torch.save({
        "model_state":   mlp.state_dict(),
        "input_dim":     X_all.shape[1],         # 4
        "bino_p5":       bino_p5,
        "bino_p95":      bino_p95,
        "feature_names": [
            "deberta_prob_cal",
            "xgb_prob_cal",
            "binoculars_ai",
            "genre_formality",
        ],
    }, mlp_path)
    log.info("MLP saved → %s", mlp_path)

    # ── Save calibrators ──────────────────────────────────────────────────────
    # inference.py MUST apply the same calibration — save both calibrators.
    cal_path = cfg.output_dir / "calibrators.pkl"
    joblib.dump({"deberta": deberta_cal, "xgb": xgb_cal}, cal_path)
    log.info("Calibrators saved → %s", cal_path)

    # ── Save eval report ──────────────────────────────────────────────────────
    report_path = cfg.output_dir / "eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Ensemble MLP — Evaluation Report\n")
        f.write("Signals: DeBERTa-v3-large | XGBoost | Binoculars\n")
        f.write("= " * 55 + "\n\n")
        for m in [train_m, val_m, test_m]:
            f.write(f"{m['split'].upper()}\n")
            for k, v in m.items():
                if k != "split":
                    f.write(f"  {k: <12}: {v:.4f}\n")
            f.write("\n")
    log.info("Report saved → %s", report_path)

    log.info(" ")
    log.info("= " * 55)
    log.info("ENSEMBLE FINAL RESULTS")
    log.info("= " * 55)
    log.info("Val  F1      : %.4f", val_m["f1"])
    log.info("Val  AUC-ROC : %.4f", val_m["auc_roc"])
    log.info("Test F1      : %.4f", test_m["f1"])
    log.info("Test AUC-ROC : %.4f", test_m["auc_roc"])
    log.info("= " * 55)
    log.info("Done!")


if __name__ == "__main__":
    run()