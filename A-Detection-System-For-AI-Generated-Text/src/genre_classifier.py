"""
Genre Classifier — Inference Wrapper
Loads the fine-tuned DistilBERT genre model and returns a single
formality_score for use as the 4th signal in the ensemble MLP.

formality_score = P(encyclopedic) + P(academic)
  → 0.0 = clearly informal/general (trust DeBERTa normally)
  → 1.0 = clearly encyclopedic/academic (DeBERTa is unreliable here)

Usage:
    from genre_classifier import GenreClassifier
    gc = GenreClassifier()
    gc.load()
    score = gc.predict("The Treaty of Versailles was signed in 1919 ...")
    # score ≈ 0.95  (encyclopedic — should reduce AI confidence)
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_GENRE_DIR     = _PROJECT_ROOT / "models" / "genre" / "best_model"

GENRE_NAMES    = ("encyclopedic", "academic", "informal", "general")
ENCYCLOPEDIC   = 0
ACADEMIC       = 1
INFORMAL       = 2
GENERAL        = 3


class GenreClassifier:
    """
    Lightweight wrapper around the fine-tuned DistilBERT genre model.
    Returns a single formality_score in [0, 1]:
        high = encyclopedic or academic text
        low  = informal or general text
    """

    def __init__(self, model_dir: Path = _GENRE_DIR):
        self.model_dir  = model_dir
        self.model      = None
        self.tokenizer  = None
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._loaded    = False

    def load(self):
        if self._loaded:
            return

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Genre model not found at {self.model_dir}. "
                f"Run train_genre.py first."
            )

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        log.info("Loading genre classifier from %s ...", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device).eval()

        total = sum(p.numel() for p in self.model.parameters())
        log.info("Genre model loaded (%s params, device=%s)",
                 f"{total:,}", self.device)
        self._loaded = True

    def predict_proba(self, text: str) -> np.ndarray:
        """
        Returns probability array of shape (4,) over:
            [encyclopedic, academic, informal, general]
        """
        if not self._loaded:
            self.load()

        enc = self.tokenizer(
            text,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)[0].float().cpu().numpy()

        return probs

    def predict(self, text: str) -> dict:
        """
        Returns full prediction dict including formality_score.
        formality_score = P(encyclopedic) + P(academic)
        """
        probs = self.predict_proba(text)
        return {
            "genre":            GENRE_NAMES[int(np.argmax(probs))],
            "probs":            probs,
            "p_encyclopedic":   float(probs[ENCYCLOPEDIC]),
            "p_academic":       float(probs[ACADEMIC]),
            "p_informal":       float(probs[INFORMAL]),
            "p_general":        float(probs[GENERAL]),
            # The key signal for the MLP:
            # high = formal text where DeBERTa is unreliable
            # low  = informal text where DeBERTa is trustworthy
            "formality_score":  float(probs[ENCYCLOPEDIC] + probs[ACADEMIC]),
        }

    def predict_batch(self, texts: list[str],
                      batch_size: int = 64) -> np.ndarray:
        """
        Batch inference. Returns formality_score array of shape (N,).
        Used by train_ensemble.py for the full dataset.
        """
        if not self._loaded:
            self.load()

        from tqdm import tqdm
        all_scores = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Genre"):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(
                batch,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).float().cpu().numpy()

            # formality_score = P(encyclopedic) + P(academic)
            scores = probs[:, ENCYCLOPEDIC] + probs[:, ACADEMIC]
            all_scores.extend(scores.tolist())

        return np.array(all_scores, dtype=np.float32)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    gc = GenreClassifier()
    gc.load()

    tests = {
        "encyclopedic": (
            "The Treaty of Versailles was signed on 28 June 1919 in the "
            "Hall of Mirrors at the Palace of Versailles. It officially ended "
            "the state of war between Germany and the Allied Powers."
        ),
        "academic": (
            "Recent meta-analyses have questioned the replication of several "
            "foundational findings in social psychology. Priming effects, once "
            "considered robust, have shown inconsistent results across laboratories."
        ),
        "informal": (
            "I burnt the rice again. Third time this week and I genuinely cannot "
            "explain it. My flatmate didn't say anything but I saw the look."
        ),
        "general": (
            "The company reported strong quarterly earnings, beating analyst "
            "expectations by 12 percent. Revenue grew 8 percent year-over-year."
        ),
    }

    print("\n── Genre Classifier Test ──────────────────────────────────────")
    print(f"{'Text type':<14} {'Predicted':<14} {'Formality':>10}  Probs")
    print("-" * 65)
    for expected, text in tests.items():
        result = gc.predict(text)
        probs  = result["probs"]
        print(
            f"{expected:<14} {result['genre']:<14} "
            f"{result['formality_score']:>10.3f}  "
            f"enc={probs[0]:.2f} acad={probs[1]:.2f} "
            f"inf={probs[2]:.2f} gen={probs[3]:.2f}"
        )
