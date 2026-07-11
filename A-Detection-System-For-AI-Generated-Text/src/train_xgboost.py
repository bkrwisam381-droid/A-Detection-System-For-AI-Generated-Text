import logging
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
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
        logging.FileHandler(_LOG_DIR / "train_xgboost.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────────
class XGBConfig:
    # Data
    train_csv = _PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv   = _PROJECT_ROOT / "data" / "splits" / "val.csv"
    test_csv  = _PROJECT_ROOT / "data" / "splits" / "test.csv"

    # Output
    model_dir         = _PROJECT_ROOT / "models" / "xgboost"
    model_path        = model_dir / "xgb_model.json"
    scaler_path       = model_dir / "scaler.pkl"
    feat_imp_path     = model_dir / "feature_importance.csv"
    pruned_feats_path = model_dir / "pruned_features.json"   # saved after pruning
    report_path       = model_dir / "eval_report.txt"
    feat_cache_dir    = _PROJECT_ROOT / "data" / "processed"

    # ── Feature pruning ───────────────────────────────────────────────────────
    # Features with XGBoost importance below this threshold are dropped.
    # On the first run (no pruned_features.json), all 39 features are used
    # and this file is created.  On the second run the pruned set is loaded.
    #
    # Why prune?
    #   Low-importance features add noise to tree splits and can hurt
    #   generalisation slightly.  More importantly, they inflate the feature
    #   vector for the MLP fusion head which only sees 5 signals — keeping
    #   only meaningful features makes the XGBoost probability cleaner.
    #
    # Expected pruned count: ~25-30 features (dropping ~10 weakest).
    importance_threshold: float = 0.01   # drop features below 1% importance

    # ── XGBoost hyperparameters ───────────────────────────────────────────────
    xgb_params = dict(
        n_estimators          = 1000,
        max_depth             = 8,
        learning_rate         = 0.03,
        subsample             = 0.9,
        colsample_bytree      = 0.9,
        min_child_weight      = 5,
        gamma                 = 0.1,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        scale_pos_weight      = 1.0,
        objective             = "binary:logistic",
        eval_metric           = ["logloss", "auc"],
        early_stopping_rounds = 50,
        random_state          = 42,
        n_jobs                = -1,
        tree_method           = "hist",
        verbosity             = 1,
    )

    seed = 42

cfg = XGBConfig()
cfg.model_dir.mkdir(parents=True, exist_ok=True)
cfg.feat_cache_dir.mkdir(parents=True, exist_ok=True)

# ── Feature column selection ─────────────────────────────────────────────────────
def get_active_features() -> list[str]:
    """
    Returns the feature list to use for this run.
    - If pruned_features.json exists: load and use the pruned set.
    - Otherwise: use the full 39-feature set (first run).
    """
    if cfg.pruned_feats_path.exists():
        with open(cfg.pruned_feats_path) as f:
            feats = json.load(f)
        log.info("Loaded pruned feature set (%d features) from %s",
                 len(feats), cfg.pruned_feats_path)
        return feats
    log.info("No pruned feature file found — using all %d features (first run).",
             len(FeatureExtractor.FEATURE_NAMES))
    return FeatureExtractor.FEATURE_NAMES

# ── Feature extraction with caching ─────────────────────────────────────────────
def load_or_extract(split: str, df: pd.DataFrame,
                    feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features for a split, using a per-feature-set cache.
    Cache key includes the number of features so changing the feature set
    automatically invalidates stale cache files.
    """
    n_feats = len(feature_cols)
    feat_cache  = cfg.feat_cache_dir / f"{split}_features_{n_feats}f.npy"
    label_cache = cfg.feat_cache_dir / f"{split}_labels.npy"

    if feat_cache.exists() and label_cache.exists():
        log.info("Loading cached features for '%s' split (%d features) ...",
                 split, n_feats)
        X = np.load(feat_cache)
        y = np.load(label_cache)
        log.info("  Loaded: X=%s  y=%s", X.shape, y.shape)
        return X, y

    log.info("Extracting features for '%s' split (%d texts, %d features) ...",
             split, len(df), n_feats)
    extractor = FeatureExtractor()
    feat_df   = extractor.extract_dataframe(
        texts  = df["text"].tolist(),
        labels = df["label"].tolist(),
        desc   = f"Features [{split}]",
    )

    X = feat_df[feature_cols].values.astype(np.float32)
    y = feat_df["label"].values.astype(np.int32)

    np.save(feat_cache,  X)
    np.save(label_cache, y)
    log.info("  Saved cache: %s  %s", feat_cache, label_cache)
    return X, y

# ── Evaluation ───────────────────────────────────────────────────────────────────
def evaluate(model, scaler, X: np.ndarray, y: np.ndarray,
             split_name: str) -> dict:
    X_s   = scaler.transform(X)
    probs = model.predict_proba(X_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "split":     split_name,
        "accuracy":  accuracy_score(y, preds),
        "f1":        f1_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "auc_roc":   roc_auc_score(y, probs),
    }
    log.info(
        "%s | acc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f  auc=%.4f",
        split_name.upper().ljust(6),
        metrics["accuracy"], metrics["f1"],
        metrics["precision"], metrics["recall"],
        metrics["auc_roc"],
    )
    return metrics

# ── Main ──────────────────────────────────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("XGBoost Stylometric Classifier Training")
    log.info("=" * 60)

    # ── Feature selection ─────────────────────────────────────────────────────
    feature_cols = get_active_features()
    is_pruned_run = cfg.pruned_feats_path.exists()

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    log.info("Loading CSVs ...")
    train_df = pd.read_csv(cfg.train_csv)
    val_df   = pd.read_csv(cfg.val_csv)
    test_df  = pd.read_csv(cfg.test_csv)
    log.info("Train: %d  Val: %d  Test: %d",
             len(train_df), len(val_df), len(test_df))

    # ── Extract features ──────────────────────────────────────────────────────
    X_train, y_train = load_or_extract("train", train_df, feature_cols)
    X_val,   y_val   = load_or_extract("val",   val_df,   feature_cols)
    X_test,  y_test  = load_or_extract("test",  test_df,  feature_cols)
    log.info("Feature matrix: train=%s  val=%s  test=%s",
             X_train.shape, X_val.shape, X_test.shape)

    # ── Scale features ────────────────────────────────────────────────────────
    log.info("Fitting StandardScaler on train features ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, cfg.scaler_path)
    log.info("Scaler saved to %s", cfg.scaler_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(" ")
    log.info("Training XGBoost (%s run) ...",
             "PRUNED" if is_pruned_run else "FIRST (all features)")
    model = xgb.XGBClassifier(**cfg.xgb_params)
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=50,
    )
    log.info("Best iteration: %d", model.best_iteration)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info(" ")
    log.info("-- Evaluation --")
    train_metrics = evaluate(model, scaler, X_train, y_train, "train")
    val_metrics   = evaluate(model, scaler, X_val,   y_val,   "val")
    test_metrics  = evaluate(model, scaler, X_test,  y_test,  "test")

    preds  = model.predict(X_test_s)
    report = classification_report(y_test, preds, target_names=["human", "AI"])
    cm     = confusion_matrix(y_test, preds)
    log.info("\nClassification report (test):\n%s", report)
    log.info("Confusion matrix (test):\n%s", cm)

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info("\nTop 15 most important features:")
    log.info("\n%s", importance.head(15).to_string(index=False))
    importance.to_csv(cfg.feat_imp_path, index=False)
    log.info("Feature importance saved to %s", cfg.feat_imp_path)

    # ── Auto-pruning (first run only) ─────────────────────────────────────────
    if not is_pruned_run:
        weak = importance[importance["importance"] < cfg.importance_threshold]
        kept = importance[importance["importance"] >= cfg.importance_threshold]

        log.info(" ")
        log.info("── Auto-pruning (threshold=%.3f) ──────────────────────────",
                 cfg.importance_threshold)
        log.info("Total features : %d", len(importance))
        log.info("Kept           : %d", len(kept))
        log.info("Pruned         : %d  %s",
                 len(weak),
                 weak["feature"].tolist() if len(weak) else "none")

        pruned_feature_list = kept["feature"].tolist()
        with open(cfg.pruned_feats_path, "w") as f:
            json.dump(pruned_feature_list, f, indent=2)
        log.info("Pruned feature list saved to %s", cfg.pruned_feats_path)
        log.info(" ")
        log.info(">>> Re-run train_xgboost.py to train on the pruned feature set.")
        log.info("    Cache will be automatically invalidated (different filename).")

    # ── Save model ────────────────────────────────────────────────────────────
    model.save_model(cfg.model_path)
    log.info("Model saved to %s", cfg.model_path)

    # ── Save feature list used ────────────────────────────────────────────────
    # Always save which features the saved model was trained on —
    # inference.py and train_ensemble.py must use the same list.
    active_feats_path = cfg.model_dir / "active_features.json"
    with open(active_feats_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info("Active feature list saved to %s", active_feats_path)

    # ── Eval report ───────────────────────────────────────────────────────────
    with open(cfg.report_path, "w", encoding="utf-8") as f:
        f.write("XGBoost Stylometric Classifier - Evaluation Report\n")
        f.write("Run type: " + ("PRUNED" if is_pruned_run else "FIRST (all features)") + "\n")
        f.write("Features used: " + str(len(feature_cols)) + "\n")
        f.write("= " * 55 + "\n\n")
        for m in [train_metrics, val_metrics, test_metrics]:
            f.write(f"{m['split'].upper()}\n")
            for k, v in m.items():
                if k != "split":
                    f.write(f"  {k: <12}: {v:.4f}\n")
            f.write("\n")
        f.write("\nClassification Report (test):\n")
        f.write(report)
        f.write("\nConfusion Matrix (test):\n")
        f.write(str(cm))
    log.info("Eval report saved to %s", cfg.report_path)

    log.info(" ")
    log.info("= " * 55)
    log.info("SUMMARY  (%d features)", len(feature_cols))
    log.info("= " * 55)
    log.info("Val  F1      : %.4f", val_metrics["f1"])
    log.info("Val  AUC-ROC : %.4f", val_metrics["auc_roc"])
    log.info("Test F1      : %.4f", test_metrics["f1"])
    log.info("Test AUC-ROC : %.4f", test_metrics["auc_roc"])
    log.info("Model        : %s",   cfg.model_path)
    if not is_pruned_run:
        log.info(" ")
        log.info("NEXT STEP: Re-run train_xgboost.py to use pruned features.")
    log.info("Done!")


if __name__ == "__main__":
    run()