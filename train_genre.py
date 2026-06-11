import logging
import warnings
import os
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
from sklearn.metrics import accuracy_score, f1_score

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
        logging.FileHandler(_LOG_DIR / "train_genre.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────────
@dataclass
class GenreConfig:
    # Model — DistilBERT is 66M params, ~6x faster than BERT, good enough for genre
    model_name: str = "distilbert-base-uncased"

    # Output
    output_dir:     Path = _PROJECT_ROOT / "models" / "genre"
    best_model_dir: Path = _PROJECT_ROOT / "models" / "genre" / "best_model"

    # Data
    samples_per_class: int = 20_000   # 80k total, balanced
    max_length:        int = 256      # genre is detectable from first 256 tokens
    val_split:         float = 0.1
    test_split:        float = 0.1

    # Training
    num_epochs:        int   = 4
    train_batch_size:  int   = 16
    eval_batch_size:   int   = 32
    learning_rate:     float = 3e-5
    warmup_ratio:      float = 0.1
    weight_decay:      float = 0.01
    grad_accumulation: int   = 2      # effective batch = 32
    eval_steps:        int   = 200
    early_stopping:    int   = 3
    fp16:              bool  = False
    bf16:              bool  = True
    seed:              int   = 42

    # Genre labels
    GENRES: tuple = ("encyclopedic", "academic", "informal", "general")
    id2label: dict = None
    label2id: dict = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.id2label = {i: g for i, g in enumerate(self.GENRES)}
        self.label2id = {g: i for i, g in enumerate(self.GENRES)}

cfg = GenreConfig()

# ── Genre labels ──────────────────────────────────────────────────────────────────
ENCYCLOPEDIC = 0
ACADEMIC     = 1
INFORMAL     = 2
GENERAL      = 3


# ══════════════════════════════════════════════════════════════════════════════
# Data loading — all from HuggingFace, no manual labeling needed
# ══════════════════════════════════════════════════════════════════════════════
def load_wikipedia(n: int) -> list[dict]:
    """Wikipedia → encyclopedic (label 0)"""
    log.info("Loading Wikipedia (encyclopedic) ...")
    from datasets import load_dataset
    # wikimedia/wikipedia is the updated dataset — no script required
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    rows = []
    for item in ds:
        text = item.get("text", "")
        # Take first paragraph — clean encyclopedic summary
        paras = [p.strip() for p in text.split("\n\n") if len(p.split()) >= 40]
        if paras:
            rows.append({"text": paras[0][:1500], "label": ENCYCLOPEDIC})
        if len(rows) >= n:
            break
    log.info("  Wikipedia: %d samples", len(rows))
    return rows


def load_arxiv(n: int) -> list[dict]:
    """arXiv abstracts → academic (label 1)"""
    log.info("Loading arXiv abstracts (academic) ...")
    from datasets import load_dataset

    rows = []

    # Primary: ccdv/arxiv-classification — abstracts only, no scripts
    try:
        ds = load_dataset(
            "ccdv/arxiv-classification",
            split="train",
            streaming=True,
        )
        for item in ds:
            abstract = item.get("abstract", "").strip()
            if len(abstract.split()) >= 40:
                rows.append({"text": abstract[:1500], "label": ACADEMIC})
            if len(rows) >= n:
                break
        log.info("  arXiv (ccdv): %d samples", len(rows))
    except Exception as e:
        log.warning("  ccdv/arxiv-classification failed: %s", e)

    # Fallback: gfissore/arxiv-abstracts-2021
    if len(rows) < n:
        log.info("  Trying fallback arXiv dataset ...")
        try:
            ds2 = load_dataset(
                "gfissore/arxiv-abstracts-2021",
                split="train",
                streaming=True,
            )
            for item in ds2:
                abstract = item.get("abstract", "").strip()
                if len(abstract.split()) >= 40:
                    rows.append({"text": abstract[:1500], "label": ACADEMIC})
                if len(rows) >= n:
                    break
            log.info("  arXiv fallback: %d samples total", len(rows))
        except Exception as e:
            log.warning("  arXiv fallback also failed: %s", e)

    log.info("  arXiv: %d samples", len(rows))
    return rows[:n]


def load_reddit(n: int) -> list[dict]:
    """Reddit posts → informal (label 2)"""
    log.info("Loading Reddit (informal) ...")
    from datasets import load_dataset

    rows = []

    # Primary: reddit_tifu long stories
    try:
        ds = load_dataset("reddit_tifu", "long", split="train", streaming=True)
        for item in ds:
            text = item.get("selftext", item.get("story", "")).strip()
            if len(text.split()) >= 40:
                rows.append({"text": text[:1500], "label": INFORMAL})
            if len(rows) >= n:
                break
        log.info("  reddit_tifu: %d samples", len(rows))
    except Exception as e:
        log.warning("  reddit_tifu failed: %s", e)

    # Fallback 1: sentence-transformers/reddit-title-body
    if len(rows) < n:
        log.info("  Trying reddit title-body fallback ...")
        try:
            ds2 = load_dataset(
                "sentence-transformers/reddit-title-body",
                split="train",
                streaming=True,
            )
            for item in ds2:
                text = item.get("body", "").strip()
                if len(text.split()) >= 40:
                    rows.append({"text": text[:1500], "label": INFORMAL})
                if len(rows) >= n:
                    break
            log.info("  reddit title-body: %d samples total", len(rows))
        except Exception as e:
            log.warning("  reddit title-body failed: %s", e)

    # Fallback 2: webtext (OpenAI WebText — informal web writing)
    if len(rows) < n:
        log.info("  Trying webtext fallback ...")
        try:
            ds3 = load_dataset("Skylion007/openwebtext", split="train",
                               streaming=True)
            for item in ds3:
                text = item.get("text", "").strip()
                words = text.split()
                if 40 <= len(words) <= 300:
                    rows.append({"text": text[:1500], "label": INFORMAL})
                if len(rows) >= n:
                    break
            log.info("  webtext: %d samples total", len(rows))
        except Exception as e:
            log.warning("  webtext failed: %s", e)

    log.info("  Informal total: %d samples", len(rows))
    return rows[:n]


def load_news(n: int) -> list[dict]:
    """News articles → general (label 3)"""
    log.info("Loading news articles (general) ...")
    from datasets import load_dataset

    rows = []

    # Primary: fancyzhx/ag_news (updated name for ag_news)
    try:
        ds = load_dataset("fancyzhx/ag_news", split="train", streaming=False)
        for item in ds:
            text = item.get("text", "").strip()
            if len(text.split()) >= 20:
                rows.append({"text": text[:1500], "label": GENERAL})
            if len(rows) >= n:
                break
        log.info("  AG News: %d samples", len(rows))
    except Exception as e:
        log.warning("  fancyzhx/ag_news failed: %s", e)
        # Try original name
        try:
            ds = load_dataset("ag_news", split="train", streaming=False)
            for item in ds:
                text = item.get("text", "").strip()
                if len(text.split()) >= 20:
                    rows.append({"text": text[:1500], "label": GENERAL})
                if len(rows) >= n:
                    break
            log.info("  AG News (original): %d samples", len(rows))
        except Exception as e2:
            log.warning("  ag_news also failed: %s", e2)

    # Fallback: cc_news
    if len(rows) < n:
        log.info("  Trying cc_news fallback ...")
        try:
            ds2 = load_dataset("cc_news", split="train", streaming=True)
            for item in ds2:
                text = item.get("text", "").strip()
                if len(text.split()) >= 40:
                    rows.append({"text": text[:1500], "label": GENERAL})
                if len(rows) >= n:
                    break
            log.info("  cc_news: %d samples total", len(rows))
        except Exception as e:
            log.warning("  cc_news failed: %s", e)

    # Fallback 2: sentence-transformers/natural-questions
    if len(rows) < n:
        log.info("  Trying natural-questions fallback ...")
        try:
            ds3 = load_dataset(
                "sentence-transformers/natural-questions",
                split="train",
                streaming=True,
            )
            for item in ds3:
                text = item.get("answer", "").strip()
                if len(text.split()) >= 40:
                    rows.append({"text": text[:1500], "label": GENERAL})
                if len(rows) >= n:
                    break
            log.info("  natural-questions: %d samples total", len(rows))
        except Exception as e:
            log.warning("  natural-questions failed: %s", e)

    log.info("  General total: %d samples", len(rows))
    return rows[:n]


def build_dataset() -> pd.DataFrame:
    """Load all 4 genres and return balanced DataFrame."""
    n = cfg.samples_per_class
    all_rows = (
        load_wikipedia(n) +
        load_arxiv(n) +
        load_reddit(n) +
        load_news(n)
    )
    df = pd.DataFrame(all_rows).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    log.info("Total samples: %d", len(df))
    for i, genre in enumerate(cfg.GENRES):
        log.info("  %s: %d", genre, (df.label == i).sum())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class GenreDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        log.info("Pre-tokenizing %d texts ...", len(df))
        texts  = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        enc = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = labels
        log.info("Tokenization done.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run():
    log.info("=" * 60)
    log.info("Genre Classifier Training (DistilBERT)")
    log.info("Classes: %s", cfg.GENRES)
    log.info("=" * 60)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Build dataset ─────────────────────────────────────────────────────────
    log.info("Building dataset ...")
    df = build_dataset()

    # Train / val / test split
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        df, test_size=cfg.val_split + cfg.test_split,
        stratify=df["label"], random_state=cfg.seed,
    )
    rel_test = cfg.test_split / (cfg.val_split + cfg.test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["label"], random_state=cfg.seed,
    )
    log.info("Train: %d  Val: %d  Test: %d",
             len(train_df), len(val_df), len(test_df))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info("Loading tokenizer: %s ...", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    train_ds = GenreDataset(train_df, tokenizer, cfg.max_length)
    val_ds   = GenreDataset(val_df,   tokenizer, cfg.max_length)
    test_ds  = GenreDataset(test_df,  tokenizer, cfg.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Loading model: %s ...", cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(cfg.GENRES),
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %s total / %s trainable",
             f"{total_params:,}", f"{trainable_params:,}")

    # ── Training args ─────────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_ds) // (
        cfg.train_batch_size * cfg.grad_accumulation
    ))
    total_steps  = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    log.info("Steps per epoch: %d  Total: %d  Warmup: %d",
             steps_per_epoch, total_steps, warmup_steps)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_dir=str(_LOG_DIR),
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping
        )],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(" ")
    log.info("Starting training ...")
    log.info("Expected: ~1-2h total")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    if torch.cuda.is_available():
        log.info("Peak VRAM: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving best model → %s", cfg.best_model_dir)
    trainer.save_model(str(cfg.best_model_dir))
    tokenizer.save_pretrained(str(cfg.best_model_dir))

    # ── Evaluate on test set ──────────────────────────────────────────────────
    log.info("Final evaluation on test set ...")
    test_result = trainer.predict(test_ds)
    test_preds  = np.argmax(test_result.predictions, axis=-1)
    test_labels = test_result.label_ids

    from sklearn.metrics import classification_report
    report = classification_report(
        test_labels, test_preds,
        target_names=list(cfg.GENRES),
    )

    log.info("\n%s", report)

    acc      = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average="macro")

    log.info("=" * 60)
    log.info("GENRE CLASSIFIER RESULTS")
    log.info("=" * 60)
    log.info("Test Accuracy : %.4f", acc)
    log.info("Test F1 Macro : %.4f", f1_macro)
    log.info("=" * 60)
    log.info("Runtime: %.1f hours", train_result.metrics["train_runtime"] / 3600)
    log.info("Model saved → %s", cfg.best_model_dir)
    log.info("Done!")
    log.info(" ")
    log.info("NEXT STEP: Run train_ensemble.py to add genre as 4th MLP signal.")


if __name__ == "__main__":
    run()