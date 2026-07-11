import os
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
        logging.FileHandler(_LOG_DIR / "train_deberta.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── GPU Diagnostics ──────────────────────────────────────────────────────────────
def log_gpu_info():
    if torch.cuda.is_available():
        gpu_name  = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("=" * 60)
        log.info("GPU            : %s", gpu_name)
        log.info("VRAM           : %.1f GB", vram_total)
        log.info("CUDA Version   : %s", torch.version.cuda)
        log.info("PyTorch Version: %s", torch.__version__)
        log.info("=" * 60)
    else:
        log.warning("NO GPU DETECTED - Training will be extremely slow!")

# ── Config ───────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    # Upgraded from deberta-v3-base (~180M) to deberta-v3-large (~400M)
    # Expect +1.5-3 F1 points at the cost of ~2.5x longer training time
    model_name: str = "microsoft/deberta-v3-large"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_csv: Path = _PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv:   Path = _PROJECT_ROOT / "data" / "splits" / "val.csv"

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:     Path = _PROJECT_ROOT / "models" / "deberta"
    best_model_dir: Path = _PROJECT_ROOT / "models" / "deberta" / "best_model"

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # 512 captures full document context. AI writing patterns (long-range
    # coherence, topic drift) are most visible at this length.
    # With batch=1 + gradient checkpointing + BF16, fits in 8GB VRAM.
    max_length: int = 448

    # ── Training hyperparameters ──────────────────────────────────────────────
    num_epochs:        int   = 4        # large model converges faster
    train_batch_size:  int   = 1        # tiny batch; gradient checkpointing compensates
    eval_batch_size:   int   = 4        # eval has no backward pass, can be larger
    learning_rate:     float = 2e-5     # lower LR for larger model (standard)
    warmup_ratio:      float = 0.1
    weight_decay:      float = 0.01
    grad_accumulation: int   = 32       # effective batch = 1 × 32 = 32

    # ── Stability / memory ────────────────────────────────────────────────────
    # gradient_checkpointing: recomputes activations during backward pass
    # instead of storing them. Saves ~40% VRAM at ~20% speed cost.
    # Safe with BF16 (was unstable with FP32 on older DeBERTa versions).
    fp16: bool = False
    bf16: bool = True   # RTX 4060 supports BF16 natively

    gradient_checkpointing_enabled: bool = False


    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping_patience: int = 7 

    # ── Evaluation frequency ──────────────────────────────────────────────────
    # More frequent than base (500) because large steps are slower —
    # we want to catch the best checkpoint sooner.
    eval_steps: int = 300

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

cfg = TrainConfig()

# ── Dataset ──────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    """Pre-tokenizes all texts at construction time — avoids tokenization
    overhead inside the DataLoader workers, which is the main speed bottleneck."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        if len(df) == 0:
            raise ValueError("Empty dataframe provided to TextDataset")

        log.info("Pre-tokenizing %d texts (max_length=%d) ...", len(df), max_length)

        texts  = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False,       # DataCollatorWithPadding handles padding per-batch
            return_tensors=None,
        )

        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = labels

        lengths = [len(ids) for ids in self.input_ids]
        log.info("Tokenization done. Avg length: %.0f  Max: %d  (budget: %d)",
                 np.mean(lengths), max(lengths), max_length)
        trunc_pct = sum(1 for l in lengths if l == max_length) / len(lengths) * 100
        log.info("Texts hitting max_length (truncated): %.1f%%", trunc_pct)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

# ── Metrics ───────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", zero_division=0),
        "auc_roc":   roc_auc_score(labels, probs[:, 1]),
    }

# ── Main ──────────────────────────────────────────────────────────────────────────
def run():
    log_gpu_info()

    log.info("=" * 60)
    log.info("DeBERTa-v3-LARGE Fine-tuning")
    log.info("Model          : %s", cfg.model_name)
    log.info("Max length     : %d tokens", cfg.max_length)
    log.info("Batch          : %d (accumulation %d → effective %d)",
             cfg.train_batch_size, cfg.grad_accumulation,
             cfg.train_batch_size * cfg.grad_accumulation)
    log.info("Grad checkpoint: ENABLED (saves ~40%% VRAM, +~20%% time)")
    log.info("Precision      : BF16")
    log.info("=" * 60)

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = False   # deterministic preferred for large runs

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading data ...")
    if not cfg.train_csv.exists():
        raise FileNotFoundError(f"Train file not found: {cfg.train_csv}")
    if not cfg.val_csv.exists():
        raise FileNotFoundError(f"Val file not found: {cfg.val_csv}")

    train_df = pd.read_csv(cfg.train_csv)
    val_df   = pd.read_csv(cfg.val_csv)

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"]   = val_df["label"].astype(int)

    log.info("Train: %d rows  (human=%d  AI=%d)",
             len(train_df), (train_df.label==0).sum(), (train_df.label==1).sum())
    log.info("Val:   %d rows  (human=%d  AI=%d)",
             len(val_df),   (val_df.label==0).sum(),   (val_df.label==1).sum())

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info("Loading tokenizer: %s ...", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    train_dataset = TextDataset(train_df, tokenizer, cfg.max_length)
    val_dataset   = TextDataset(val_df,   tokenizer, cfg.max_length)

    # pad_to_multiple_of=8 — required for BF16 tensor cores on Ampere+
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Loading model: %s ...", cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
    )

    # Enable gradient checkpointing BEFORE moving to device.
    # This is safe with BF16. Was unstable in FP32 due to DeBERTa's disentangled
    # attention mechanism, which is no longer an issue here.
 

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %s total / %s trainable",
             f"{total_params:,}", f"{trainable_params:,}")

    # ── Training schedule ─────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_dataset) // (
        cfg.train_batch_size * cfg.grad_accumulation
    ))
    total_steps  = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    log.info("Training schedule:")
    log.info("  Steps per epoch : %d", steps_per_epoch)
    log.info("  Total steps     : %d", total_steps)
    log.info("  Warmup steps    : %d", warmup_steps)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),

        # Batch settings
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,

        # Optimizer
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        # Gradient checkpointing
        # NOTE: set here AND called model.gradient_checkpointing_enable() above
        # because some HF versions need one or the other.
        gradient_checkpointing=False,  # set to False here; enable via model method for better compatibility

        # Evaluation
        eval_delay=warmup_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Precision
        fp16=cfg.fp16,
        bf16=cfg.bf16,

        # Logging
        logging_dir=str(_LOG_DIR),
        logging_steps=50,
        report_to="none",

        # Reproducibility
        seed=cfg.seed,
        data_seed=cfg.seed,

        # Data loading
        # 0 workers on Windows — avoids multiprocessing spawn issues
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        ],
    )

    # ── VRAM check ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        log.info("VRAM before training: %.2f GB", torch.cuda.memory_allocated() / 1e9)

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(" ")
    log.info("Starting training ...")
    log.info("Expected: ~5-10s/it (large model), ~15-25 hours total")
    log.info("If OOM: reduce max_length to 384 or reduce grad_accumulation to 16")
    log.info(" ")

    train_result = trainer.train()

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        log.info("Peak VRAM usage: %.2f GB / 8.00 GB", peak_vram)
        if peak_vram > 7.5:
            log.warning("Very close to VRAM limit! Consider reducing max_length to 384.")

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving best model to %s ...", cfg.best_model_dir)
    trainer.save_model(str(cfg.best_model_dir))
    tokenizer.save_pretrained(str(cfg.best_model_dir))

    # ── Final eval ────────────────────────────────────────────────────────────
    log.info(" ")
    log.info("Final evaluation on validation set ...")
    metrics = trainer.evaluate()

    log.info(" ")
    log.info("=" * 60)
    log.info("FINAL RESULTS  (deberta-v3-large, max_len=%d)", cfg.max_length)
    log.info("=" * 60)
    for key in ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_auc_roc"]:
        log.info("%-12s: %.4f", key.replace("eval_", ""), metrics.get(key, 0))
    log.info("=" * 60)

    # ── Save training log ─────────────────────────────────────────────────────
    log_path = cfg.output_dir / "training_log.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(log_path, index=False)

    log.info(" ")
    log.info("Runtime : %.1f hours", train_result.metrics["train_runtime"] / 3600)
    log.info("Model   : %s", cfg.best_model_dir)
    log.info("Done!")


if __name__ == "__main__":
    run()