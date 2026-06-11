import os
import re
import json
import logging
import hashlib
import gzip
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# langdetect is lightweight (~1ms per text) and accurate enough for filtering
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ── Windows-safe logging (ASCII only, utf-8 file) ─────────────────────────────
def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "data").exists():
            return candidate
    return here

_PROJECT_ROOT = _find_project_root()
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    # Sample budget - Increased to 120k to include modern AI models
    total_samples: int = 240_000
    
    # Text quality filters
    min_words: int = 50
    max_words: int = 512
    min_unique_ratio: float = 0.20

    # Split ratios
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15

    seed: int = 42

    # Paths
    hc3_dir:  Path = None
    raid_dir: Path = None
    wildchat_dir: Path = None
    anthropic_dir: Path = None
    split_dir: Path = None
    log_dir:  Path = None

    def __post_init__(self):
        if self.hc3_dir  is None: self.hc3_dir  = _PROJECT_ROOT / "data" / "raw" / "hc3"
        if self.raid_dir is None: self.raid_dir = _PROJECT_ROOT / "data" / "raw" / "raid"
        if self.wildchat_dir is None: self.wildchat_dir = _PROJECT_ROOT / "data" / "raw" / "wildchat"
        if self.anthropic_dir is None: self.anthropic_dir = _PROJECT_ROOT / "data" / "raw" / "anthropic"
        if self.split_dir is None: self.split_dir = _PROJECT_ROOT / "data" / "splits"
        if self.log_dir  is None: self.log_dir  = _PROJECT_ROOT / "logs"
        
        for d in [self.hc3_dir, self.raid_dir, self.wildchat_dir, 
                  self.anthropic_dir, self.split_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        log.info("Project root : %s", _PROJECT_ROOT)
        log.info("HC3 dir      : %s", self.hc3_dir)
        log.info("RAID dir     : %s", self.raid_dir)
        log.info("WildChat dir : %s", self.wildchat_dir)
        log.info("Anthropic dir: %s", self.anthropic_dir)
        log.info("Output dir   : %s", self.split_dir)
        
        self.n_human = self.total_samples // 2
        self.n_ai    = self.total_samples - self.n_human

cfg = PipelineConfig()

# ── Text cleaning ───────────────────────────────────────────────────────────────
def is_english(text: str) -> bool:
    if not LANGDETECT_AVAILABLE:
        return True
    try:
        return detect(text[:500]) == "en"
    except LangDetectException:
        return True

def clean_text(text) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)

    # Language filter -- English only
    if not is_english(text):
        return None

    words = text.split()
    n = len(words)

    if n < cfg.min_words or n > cfg.max_words:
        return None

    if len(set(w.lower() for w in words)) / n < cfg.min_unique_ratio:
        return None

    return text

def fingerprint(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()

# ── HC3 loader ──────────────────────────────────────────────────────────────────
def load_hc3() -> pd.DataFrame:
    files = list(cfg.hc3_dir.glob("*.jsonl")) + list(cfg.hc3_dir.glob("*.json"))
    
    if not files:
        log.warning("HC3 not found in %s -- skipping.", cfg.hc3_dir)
        return pd.DataFrame(columns=["text", "label", "source", "model"])

    rows = []
    for fpath in files:
        log.info("Reading HC3: %s", fpath.name)
        with open(fpath, encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="HC3"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            for ans in item.get("human_answers", []):
                c = clean_text(ans)
                if c:
                    rows.append({"text": c, "label": 0, "source": "hc3", "model": "human"})

            for ans in item.get("chatgpt_answers", []):
                c = clean_text(ans)
                if c:
                    rows.append({"text": c, "label": 1, "source": "hc3", "model": "chatgpt"})

    df = pd.DataFrame(rows)
    log.info("HC3: %d samples (%d human / %d AI)", len(df), (df.label==0).sum(), (df.label==1).sum())
    return df

# ── RAID loader ─────────────────────────────────────────────────────────────────
def _detect_columns(columns):
    text_col  = next((c for c in columns if c.lower() in ("generation", "text", "content")), None)
    model_col = next((c for c in columns if c.lower() == "model"), None)
    return text_col, model_col

def _process_raid_chunk(chunk: pd.DataFrame, text_col: str, model_col: str) -> list:
    rows = []
    for _, row in chunk.iterrows():
        model = str(row[model_col]).strip() if model_col else "unknown"
        label = 0 if model.lower() == "human" else 1
        c = clean_text(row[text_col])
        if c:
            rows.append({"text": c, "label": label, "source": "raid", "model": model})
    return rows

def load_raid() -> pd.DataFrame:
    files = sorted(cfg.raid_dir.glob("*.parquet")) + sorted(cfg.raid_dir.glob("*.csv"))

    if not files:
        log.warning("RAID not found in %s -- skipping.", cfg.raid_dir)
        return pd.DataFrame(columns=["text", "label", "source", "model"])

    log.info("Found %d RAID file(s) in %s", len(files), cfg.raid_dir)

    soft_cap = cfg.total_samples * 2
    all_rows = []
    text_col = model_col = None

    for fpath in files:
        if len(all_rows) >= soft_cap:
            log.info("Soft cap reached (%d rows) -- skipping remaining shards", len(all_rows))
            break

        log.info("Processing shard: %s (%.1f MB)", fpath.name, fpath.stat().st_size / 1_048_576)
        try:
            if fpath.suffix == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(fpath)

                if text_col is None:
                    cols = pf.schema_arrow.names
                    text_col, model_col = _detect_columns(cols)
                    if text_col is None:
                        log.error("RAID parquet: no text column. Columns: %s", cols)
                        continue
                    log.info("RAID columns -> text: '%s' model: '%s'", text_col, model_col)

                for batch in pf.iter_batches(batch_size=10_000, columns=[c for c in [text_col, model_col] if c]):
                    chunk = batch.to_pandas()
                    all_rows.extend(_process_raid_chunk(chunk, text_col, model_col))
                    if len(all_rows) >= soft_cap:
                        break

            else:
                CHUNK = 10_000
                reader = pd.read_csv(fpath, chunksize=CHUNK, low_memory=False)
                for i, chunk in enumerate(reader):
                    if i == 0 and text_col is None:
                        text_col, model_col = _detect_columns(list(chunk.columns))
                        if text_col is None:
                            log.error("RAID CSV: no text column. Columns: %s", list(chunk.columns))
                            break
                        log.info("RAID columns -> text: '%s' model: '%s'", text_col, model_col)
                    all_rows.extend(_process_raid_chunk(chunk, text_col, model_col))
                    if len(all_rows) >= soft_cap:
                        break

        except Exception as e:
            log.warning("Failed to read %s: %s", fpath.name, e)
            continue

        log.info(" -> running total: %d rows", len(all_rows))

    if not all_rows:
        log.error("RAID produced no usable rows.")
        return pd.DataFrame(columns=["text", "label", "source", "model"])

    df = pd.DataFrame(all_rows)
    log.info("RAID loaded: %d samples (%d human / %d AI)", len(df), (df.label==0).sum(), (df.label==1).sum())
    return df

# ── WildChat loader (UPDATED for multi-file) ────────────────────────────────────
def load_wildchat() -> pd.DataFrame:
    files = list(cfg.wildchat_dir.glob("*.parquet")) + list(cfg.wildchat_dir.glob("*.csv"))
    
    if not files:
        log.warning("WildChat not found in %s -- skipping.", cfg.wildchat_dir)
        return pd.DataFrame(columns=["text", "label", "source", "model"])

    log.info("Loading WildChat (GPT-4) samples...")
    log.info("Found %d WildChat file(s)", len(files))
    
    rows = []
    for fpath in files:
        log.info("Reading WildChat: %s", fpath.name)
        
        try:
            df_raw = pd.read_parquet(fpath)
            
            # Filter for GPT-4 responses only
            df_gpt4 = df_raw[df_raw["model"] == "gpt-4"].head(15000)
            
            for _, row in tqdm(df_gpt4.iterrows(), total=len(df_gpt4), desc=f"WildChat {fpath.name}"):
                conversation = row.get("conversation", [])
                if isinstance(conversation, str):
                    try:
                        conversation = json.loads(conversation)
                    except:
                        continue
                
                if len(conversation) > 0:
                    if isinstance(conversation[0], dict):
                        text = conversation[0].get("content", "")
                    else:
                        text = str(conversation[0])
                    
                    c = clean_text(text)
                    if c:
                        rows.append({"text": c, "label": 1, "source": "wildchat", "model": "gpt-4"})
        
        except Exception as e:
            log.warning("Failed to load WildChat %s: %s", fpath.name, e)
            continue

    df = pd.DataFrame(rows)
    log.info("WildChat: %d GPT-4 samples", len(df))
    return df

# ── Anthropic HH-RLHF loader (UPDATED for .jsonl.gz) ────────────────────────────
def load_anthropic() -> pd.DataFrame:
    files = list(cfg.anthropic_dir.glob("*.jsonl.gz")) + list(cfg.anthropic_dir.glob("*.jsonl"))
    
    if not files:
        log.warning("Anthropic HH-RLHF not found in %s -- skipping.", cfg.anthropic_dir)
        return pd.DataFrame(columns=["text", "label", "source", "model"])

    log.info("Loading Anthropic HH-RLHF (Claude) samples...")
    log.info("Found %d Anthropic file(s)", len(files))
    
    rows = []
    for fpath in files:
        log.info("Reading Anthropic: %s", fpath.name)
        
        try:
            # Handle gzipped JSONL files
            if fpath.suffix == ".gz":
                with gzip.open(fpath, "rt", encoding="utf-8") as f:
                    lines = f.readlines()
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            for line in tqdm(lines, desc=f"Anthropic {fpath.name}"):
                try:
                    item = json.loads(line.strip())
                    
                    # Extract assistant response from "chosen" field
                    chosen = item.get("chosen", "")
                    
                    if isinstance(chosen, str):
                        if "Assistant:" in chosen:
                            assistant_part = chosen.split("Assistant:")[-1].strip()
                            c = clean_text(assistant_part)
                            if c:
                                rows.append({
                                    "text": c, 
                                    "label": 1,
                                    "source": "anthropic", 
                                    "model": "claude"
                                })
                except json.JSONDecodeError:
                    continue
        
        except Exception as e:
            log.warning("Failed to load Anthropic %s: %s", fpath.name, e)
            continue

    df = pd.DataFrame(rows)
    log.info("Anthropic: %d Claude samples", len(df))
    return df

# ── Wikipedia loader (encyclopedic human text) ──────────────────────────────────
def load_wikipedia_human() -> pd.DataFrame:
    """
    Loads encyclopedic human text from Wikipedia via HuggingFace datasets.
    This directly fixes the model's blind spot for encyclopedic writing.
    Uses streaming so only pulls what it needs — no full 20GB download.
    Targets 20k clean paragraphs (first paragraph of each article = summary).
    """
    log.info("Loading Wikipedia (encyclopedic human text) ...")
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("datasets library not installed — skipping Wikipedia.")
        log.warning("Run: pip install datasets")
        return pd.DataFrame(columns=["text", "label", "source", "model"])

    rows = []
    target = 20_000

    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )
        for item in ds:
            text = item.get("text", "")
            # First paragraph = clean encyclopedic summary
            paras = [p.strip() for p in text.split("\n\n")
                     if len(p.split()) >= 50]
            if paras:
                c = clean_text(paras[0])
                if c:
                    rows.append({
                        "text":   c,
                        "label":  0,
                        "source": "wikipedia",
                        "model":  "human",
                    })
            if len(rows) >= target:
                break
    except Exception as e:
        log.warning("Wikipedia loading failed: %s", e)

    df = pd.DataFrame(rows)
    log.info("Wikipedia: %d human samples", len(df))
    return df


# ── arXiv loader (academic human text) ──────────────────────────────────────────
def load_arxiv_human() -> pd.DataFrame:
    """
    Loads academic human text from arXiv abstracts via HuggingFace datasets.
    Fixes the model's blind spot for academic/scientific writing.
    Abstracts are clean, self-contained, and clearly human-authored.
    Targets 20k abstracts.
    """
    log.info("Loading arXiv abstracts (academic human text) ...")
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("datasets library not installed — skipping arXiv.")
        return pd.DataFrame(columns=["text", "label", "source", "model"])

    rows = []
    target = 20_000

    # Try primary source first
    sources = [
        ("ccdv/arxiv-classification", "abstract"),
        ("gfissore/arxiv-abstracts-2021", "abstract"),
    ]

    for dataset_name, text_field in sources:
        if len(rows) >= target:
            break
        try:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            for item in ds:
                text = item.get(text_field, "").strip()
                c = clean_text(text)
                if c:
                    rows.append({
                        "text":   c,
                        "label":  0,
                        "source": "arxiv",
                        "model":  "human",
                    })
                if len(rows) >= target:
                    break
            log.info("  %s: loaded %d so far", dataset_name, len(rows))
        except Exception as e:
            log.warning("  %s failed: %s", dataset_name, e)

    df = pd.DataFrame(rows)
    log.info("arXiv: %d human samples", len(df))
    return df


# ── Helpers ─────────────────────────────────────────────────────────────────────
def balanced_sample(df: pd.DataFrame, n_total: int) -> pd.DataFrame:
    half  = n_total // 2
    human = df[df.label == 0]
    ai    = df[df.label == 1]
    
    n_each = min(half, len(human), len(ai))

    if n_each < half:
        log.info("Capping both classes at %d to maintain 50/50 balance (human available: %d, AI available: %d, wanted: %d each)",
                 n_each, len(human), len(ai), half)

    return pd.concat([
        human.sample(n_each, random_state=cfg.seed),
        ai.sample(n_each,    random_state=cfg.seed),
    ]).reset_index(drop=True)

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df["_fp"] = df["text"].apply(fingerprint)
    df = df.drop_duplicates(subset="_fp").drop(columns="_fp")
    log.info("Dedup: removed %d duplicates -> %d remain", before - len(df), len(df))
    return df.reset_index(drop=True)

def make_splits(df):
    train, temp = train_test_split(
        df, test_size=(cfg.val_ratio + cfg.test_ratio),
        stratify=df["label"], random_state=cfg.seed,
    )
    rel_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val, test = train_test_split(
        temp, test_size=rel_test,
        stratify=temp["label"], random_state=cfg.seed,
    )
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))

def report(name: str, df: pd.DataFrame):
    n_h = (df.label == 0).sum()
    n_a = (df.label == 1).sum()
    log.info("%-8s | total=%6d | human=%6d (%.1f%%) | AI=%6d (%.1f%%)",
             name, len(df), n_h, 100*n_h/len(df), n_a, 100*n_a/len(df))

# ── Main ────────────────────────────────────────────────────────────────────────
def run():
    log.info("=" * 50)
    log.info("AI Detector - Data Pipeline")
    log.info("Target: %s samples (%s human / %s AI)",
             f"{cfg.total_samples:,}", f"{cfg.n_human:,}", f"{cfg.n_ai:,}")
    log.info("=" * 50)
    
    frames = []
    
    # Load all datasets including WildChat and Anthropic
    for loader, name in [
        (load_hc3,              "HC3 (ChatGPT)"),
        (load_raid,             "RAID (multi-model)"),
        (load_wildchat,         "WildChat (GPT-4)"),
        (load_anthropic,        "Anthropic (Claude)"),
        (load_wikipedia_human,  "Wikipedia (encyclopedic human)"),
        (load_arxiv_human,      "arXiv (academic human)"),
    ]:
        log.info("\n[%s]", name)
        df = loader()
        if len(df):
            frames.append(df)

    if not frames:
        log.error("No data loaded! Place your files in the correct directories.")
        return

    combined = pd.concat(frames, ignore_index=True)
    log.info("\nCombined raw: %d samples", len(combined))

    combined = deduplicate(combined)
    combined = balanced_sample(combined, cfg.total_samples)
    combined = combined.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    log.info("Final dataset: %d samples", len(combined))

    train_df, val_df, test_df = make_splits(combined)

    train_df.to_csv(cfg.split_dir / "train.csv", index=False)
    val_df.to_csv(cfg.split_dir   / "val.csv",   index=False)
    test_df.to_csv(cfg.split_dir  / "test.csv",  index=False)

    log.info("\n-- Split summary --")
    report("TRAIN", train_df)
    report("VAL",   val_df)
    report("TEST",  test_df)

    log.info("\n-- AI model coverage --")
    for model, n in combined[combined.label==1]["model"].value_counts().items():
        log.info(" %-30s %d", model, n)

    log.info("\n-- Human source coverage --")
    for source, n in combined[combined.label==0]["source"].value_counts().items():
        log.info(" %-30s %d", source, n)

    log.info("\nSaved to data/splits/")
    log.info(" train.csv %d rows", len(train_df))
    log.info(" val.csv %d rows", len(val_df))
    log.info(" test.csv %d rows", len(test_df))
    log.info("\n" + "=" * 50)
    log.info("Dataset expansion summary:")
    log.info(" Old size : 200,000 samples")
    log.info(" New size : 240,000 samples (+20%%)")
    log.info(" Added    : Wikipedia (encyclopedic) + arXiv (academic) as human")
    log.info(" Key fix  : model now sees formal factual human writing in training")
    log.info(" Expected : major reduction in false positives on encyclopedic/academic text")
    log.info("=" * 50)
    log.info("Done!")

if __name__ == "__main__":
    run()