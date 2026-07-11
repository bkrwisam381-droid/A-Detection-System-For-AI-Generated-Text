"""
AI Text Detector — Inference
Pipeline: DeBERTa-v3-large + XGBoost + Binoculars + Ensemble MLP

Fixes applied (2025):
  1. predict_sentences() now uses DeBERTa only — XGBoost and Binoculars
     are unreliable at sentence level (stylometric features collapse,
     perplexity ratio is noisy on <50 tokens).  The heatmap is therefore
     more accurate for long structured documents.

  2. strip_structure() removes structural elements (TOC entries, chapter
     headers, numbered section titles, figure captions, reference list
     lines) before document-level scoring.  These elements saturate
     DeBERTa on theses/reports even when the prose itself is human-written.
     Applied only to multi-paragraph documents (>30 newlines).

  3. run_analysis() passes clean prose to models.predict() but the
     original raw text to the heatmap, so structural stripping never
     affects sentence-level colouring.
"""
import sys
import re
import json
import logging
import warnings
from pathlib import Path

# ── Rich UI ──────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.rule import Rule
    from rich.align import Align
    from rich import box
except ImportError:
    print("Rich not installed. Run: pip install rich")
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# Fusion MLP  — defined locally so inference never depends on train_ensemble.py
# Architecture is read from the checkpoint so it always matches what was saved.
# ══════════════════════════════════════════════════════════════════════════════
class FusionMLP(nn.Module):
    def __init__(self, input_dim: int = 3):
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
# Model loader
# ══════════════════════════════════════════════════════════════════════════════
class DetectorModels:
    def __init__(self):
        self.deberta_model      = None
        self.deberta_tokenizer  = None
        self.xgb_model          = None
        self.xgb_scaler         = None
        self.xgb_feature_cols   = None
        self.feature_extractor  = None
        self.bino_observer      = None   # gpt2-large
        self.bino_scorer        = None   # gpt2-xl
        self.bino_tokenizer     = None
        self.genre_classifier   = None
        self.mlp                = None
        self.mlp_meta           = None
        self.deberta_calibrator = None
        self.xgb_calibrator     = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            AutoModelForCausalLM,
        )
        import xgboost as xgb
        import joblib
        from features import FeatureExtractor

        deberta_dir      = _PROJECT_ROOT / "models" / "deberta" / "best_model"
        xgb_model_path   = _PROJECT_ROOT / "models" / "xgboost" / "xgb_model.json"
        xgb_scaler_path  = _PROJECT_ROOT / "models" / "xgboost" / "scaler.pkl"
        xgb_active_feats = _PROJECT_ROOT / "models" / "xgboost" / "active_features.json"
        mlp_path         = _PROJECT_ROOT / "models" / "ensemble" / "mlp_fusion.pt"
        cal_path         = _PROJECT_ROOT / "models" / "ensemble" / "calibrators.pkl"

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            console=console, transient=True,
        ) as prog:

            # ── DeBERTa ───────────────────────────────────────────────────────
            t = prog.add_task("Loading DeBERTa-v3-large ...", total=None)
            self.deberta_tokenizer = AutoTokenizer.from_pretrained(
                str(deberta_dir)
            )
            self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
                str(deberta_dir),
                dtype=torch.bfloat16,   # must match training dtype — avoids logit saturation
            ).to(self.device).eval()
            prog.update(t, description="[green]DeBERTa-v3-large loaded (BF16)")

            # ── XGBoost ───────────────────────────────────────────────────────
            t = prog.add_task("Loading XGBoost ...", total=None)
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_model_path)
            self.xgb_scaler        = joblib.load(xgb_scaler_path)
            self.feature_extractor = FeatureExtractor()
            if xgb_active_feats.exists():
                with open(xgb_active_feats) as f:
                    self.xgb_feature_cols = json.load(f)
            else:
                self.xgb_feature_cols = FeatureExtractor.FEATURE_NAMES
            prog.update(t, description=f"[green]XGBoost loaded "
                                       f"({len(self.xgb_feature_cols)} features)")

            # ── Binoculars ────────────────────────────────────────────────────
            t = prog.add_task("Loading Binoculars (gpt2-large + gpt2-xl) ...",
                              total=None)
            try:
                self.bino_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.bino_tokenizer.pad_token = self.bino_tokenizer.eos_token

                self.bino_observer = AutoModelForCausalLM.from_pretrained(
                    "gpt2-large", torch_dtype=torch.float16,
                ).to(self.device).eval()

                self.bino_scorer = AutoModelForCausalLM.from_pretrained(
                    "gpt2-xl", torch_dtype=torch.float16,
                ).to(self.device).eval()

                prog.update(t, description="[green]Binoculars loaded "
                                           "(gpt2-large + gpt2-xl)")
            except Exception as e:
                console.print(f"[yellow]Binoculars unavailable: {e}  "
                              f"(perplexity signal will be neutral)[/yellow]")

            # ── Ensemble MLP ──────────────────────────────────────────────────
            t = prog.add_task("Loading ensemble MLP ...", total=None)
            ckpt = torch.load(mlp_path, map_location=self.device,
                              weights_only=False)
            # Always use input_dim from the checkpoint — avoids architecture mismatch
            self.mlp = FusionMLP(input_dim=ckpt["input_dim"]).to(self.device)
            self.mlp.load_state_dict(ckpt["model_state"])
            self.mlp.eval()
            self.mlp_meta = ckpt
            prog.update(t, description=f"[green]Ensemble MLP loaded "
                                       f"({ckpt['input_dim']} signals)")

            # ── Calibrators ───────────────────────────────────────────────────
            if cal_path.exists():
                cals = joblib.load(cal_path)
                self.deberta_calibrator = cals["deberta"]
                self.xgb_calibrator     = cals["xgb"]
            else:
                console.print(
                    "[yellow]Warning: calibrators.pkl not found. "
                    "Re-run train_ensemble.py.[/yellow]"
                )

            # ── Genre classifier ──────────────────────────────────────────────
            t = prog.add_task("Loading genre classifier ...", total=None)
            try:
                from genre_classifier import GenreClassifier
                self.genre_classifier = GenreClassifier()
                self.genre_classifier.load()
                prog.update(t, description="[green]Genre classifier loaded (DistilBERT)")
            except Exception as e:
                console.print(
                    f"[yellow]Genre classifier unavailable: {e}  "
                    f"(run train_genre.py first)[/yellow]"
                )

        self._loaded = True

    # ── Per-text inference ────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """Run full pipeline on a single text. Returns result dict."""

        # ── DeBERTa ───────────────────────────────────────────────────────────
        enc = self.deberta_tokenizer(
            text, max_length=512, truncation=True,
            padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.deberta_model(**enc).logits
            deberta_prob = torch.softmax(logits, dim=-1)[0, 1].item()

        # ── XGBoost ───────────────────────────────────────────────────────────
        import pandas as _pd
        from features import FeatureExtractor as _FE
        feat_full = self.feature_extractor.extract(text)
        feat_df   = _pd.DataFrame([feat_full], columns=_FE.FEATURE_NAMES)
        feat_s    = self.xgb_scaler.transform(
            feat_df[self.xgb_feature_cols].values
        )
        xgb_prob = float(self.xgb_model.predict_proba(feat_s)[0, 1])

        # ── Binoculars ────────────────────────────────────────────────────────
        bino_raw = 1.0   # neutral fallback if models unavailable
        if self.bino_observer is not None and self.bino_scorer is not None:
            try:
                enc_b = self.bino_tokenizer(
                    text, max_length=512, truncation=True,
                    return_tensors="pt", padding=False,
                ).to(self.device)
                if enc_b["input_ids"].shape[1] >= 2:
                    with torch.no_grad():
                        ce_obs   = self.bino_observer(
                            **enc_b, labels=enc_b["input_ids"]
                        ).loss.item()
                        ce_score = self.bino_scorer(
                            **enc_b, labels=enc_b["input_ids"]
                        ).loss.item()
                    bino_raw = ce_obs / max(ce_score, 1e-6)
            except Exception:
                pass

        # Normalise using training-time percentiles from checkpoint
        bino_p5  = self.mlp_meta.get("bino_p5",  0.8)
        bino_p95 = self.mlp_meta.get("bino_p95", 1.5) + 1e-6
        bino_norm      = max(0.0, min(1.0, (bino_raw - bino_p5) / (bino_p95 - bino_p5)))
        bino_ai_signal = 1.0 - bino_norm   # invert: low ratio → high AI signal

        # ── Calibrate ─────────────────────────────────────────────────────────
        if self.deberta_calibrator is not None:
            # Clip DeBERTa at 0.99 before calibration.
            # The calibrator maps raw 1.00 → 0.98 but raw 0.99 → 0.51 — a massive
            # cliff caused by softmax saturation. DeBERTa frequently saturates at
            # exactly 1.0 on formal text, so without this clip it effectively becomes
            # a binary signal that overrides XGBoost and Binoculars.
            deberta_prob_cal = float(
                self.deberta_calibrator.predict([min(deberta_prob, 0.990)])[0]
            )
            xgb_prob_cal = float(
                self.xgb_calibrator.predict([xgb_prob])[0]
            )
        else:
            deberta_prob_cal = deberta_prob
            xgb_prob_cal     = xgb_prob

        # ── Genre classification ──────────────────────────────────────────────
        genre_formality = 0.5   # neutral fallback
        genre_info      = {}
        if self.genre_classifier is not None:
            try:
                genre_result    = self.genre_classifier.predict(text)
                genre_formality = genre_result["formality_score"]
                genre_info      = genre_result
            except Exception:
                pass

        # ── MLP fusion ────────────────────────────────────────────────────────
        # Build signal vector based on input_dim saved in checkpoint
        # input_dim=4: DeBERTa + XGBoost + Binoculars + Genre
        # input_dim=3: DeBERTa + XGBoost + Binoculars (old checkpoint)
        # input_dim=2: DeBERTa + XGBoost only (older checkpoint)
        input_dim = self.mlp_meta.get("input_dim", 4)
        if input_dim == 4:
            signals = np.array(
                [[deberta_prob_cal, xgb_prob_cal, bino_ai_signal, genre_formality]],
                dtype=np.float32,
            )
        elif input_dim == 3:
            signals = np.array(
                [[deberta_prob_cal, xgb_prob_cal, bino_ai_signal]],
                dtype=np.float32,
            )
        else:
            signals = np.array(
                [[deberta_prob_cal, xgb_prob_cal]],
                dtype=np.float32,
            )

        with torch.no_grad():
            final_prob = self.mlp(
                torch.tensor(signals).to(self.device)
            ).item()

        return {
            "final_prob":       final_prob,
            "label":            "AI" if final_prob >= 0.5 else "Human",
            "confidence":       final_prob if final_prob >= 0.5 else 1 - final_prob,
            "deberta_prob":     deberta_prob,
            "xgb_prob":         xgb_prob,
            "bino_raw":         bino_raw,
            "bino_ai_signal":   bino_ai_signal,
            "genre_formality":  genre_formality,
            "genre_info":       genre_info,
            "features":         feat_full,
        }

    # ── FIX 1: DeBERTa-only sentence scoring ─────────────────────────────────
    def predict_sentences(self, text: str) -> list[dict]:
        """
        Sentence-level predictions for the heatmap.

        Uses DeBERTa ONLY — XGBoost stylometric features are meaningless
        on a single sentence (burstiness, paragraph structure, etc. all
        collapse), and Binoculars perplexity ratio is noisy below ~50
        tokens.  Running the full ensemble pipeline on each sentence was
        the root cause of heatmap vs. document-level score conflicts on
        long structured documents (theses, reports).

        DeBERTa alone reads each sentence's semantic content correctly and
        produces reliable human/AI colouring at sentence granularity.
        """
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                 if len(s.strip()) > 10]
        results = []
        for s in sents:
            enc = self.deberta_tokenizer(
                s, max_length=512, truncation=True,
                padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.deberta_model(**enc).logits
                prob   = torch.softmax(logits, dim=-1)[0, 1].item()

            # NOTE: do NOT apply the document-level calibrator here.
            # The calibrator was trained to fix DeBERTa's softmax-saturation
            # cliff on full documents (raw 1.00 → 0.98, raw 0.99 → 0.51).
            # That cliff never occurs at sentence level — sentences produce
            # softer probabilities (0.50–0.75) because there is less context.
            # Applying the calibrator to those scores maps them below the 0.65
            # AI threshold, making the heatmap permanently blind to AI sentences.

            results.append({
                "sentence":   s,
                "prob":       prob,
                "label":      "AI" if prob >= 0.5 else "Human",
                "confidence": prob if prob >= 0.5 else 1 - prob,
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: Structural element stripper
# ══════════════════════════════════════════════════════════════════════════════
def strip_structure(text: str) -> str:
    """
    Remove lines that are likely structural rather than prose before
    passing to the document-level ensemble.  Targets:
      - Very short lines (headers, section numbers, page numbers)
      - Numbered section titles:  "1.2  Introduction",  "3.4.1  Methods"
      - Reference / citation lines:  "[1] Author et al. ..."
      - Figure / Table captions:  "Figure 3: ..."
      - TOC dot-leader lines:  "Introduction ........... 4"
      - Lines that are purely numeric or contain only punctuation

    Only applied when the document has more than 30 newlines (i.e. it
    looks like a multi-section document rather than a short paste).
    Prose lines are re-joined into a single string to preserve context
    for DeBERTa's 512-token window.
    """
    lines = text.split('\n')
    prose = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Too short to be a prose sentence
        if len(s) < 20:
            continue
        # Numbered section headings: "1.", "1.2", "1.2.3" followed by text
        if re.match(r'^[\d]{1,2}(\.[\d]{1,2}){0,3}\.?\s+[A-Z]', s):
            continue
        # Reference list entries: "[1]", "(1)", "1." at start
        if re.match(r'^(\[?\d+\]\.?|\(\d+\))\s+[A-Z]', s):
            continue
        # Figure / Table / Appendix captions
        if re.match(r'^(Figure|Fig\.|Table|Tbl\.|Appendix|Exhibit)\s', s, re.I):
            continue
        # TOC dot-leader lines (3+ consecutive dots or ellipsis chars)
        if re.search(r'\.{3,}|\u2026{2,}', s):
            continue
        # Lines that are purely numeric, Roman numerals, or page markers
        if re.match(r'^[IVXivx\d\s\-–—]+$', s):
            continue
        # Abstract / chapter / section keyword-only lines (all-caps headings)
        if re.match(r'^[A-Z][A-Z\s\-]{4,}$', s):
            continue
        prose.append(s)

    return ' '.join(prose)


# ══════════════════════════════════════════════════════════════════════════════
# File / URL extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            import fitz
            doc   = fitz.open(str(path))
            pages = []
            for page in doc:
                blocks = page.get_text("blocks")   # returns list of text blocks with coordinates
                # sort top-to-bottom, left-to-right
                blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))
                page_text = "\n".join(
                    b[4].strip() for b in blocks
                    if b[4].strip() and len(b[4].strip()) > 5
                )
                pages.append(page_text)
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            console.print("[red]PyMuPDF not installed: pip install pymupdf[/]")
            return ""
    elif suffix == ".docx":
        try:
            import docx
            return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        except ImportError:
            console.print("[red]python-docx not installed: pip install python-docx[/]")
            return ""
    else:
        console.print(f"[red]Unsupported file type: {suffix}[/]")
        return ""


def extract_text_from_url(url: str) -> str:
    try:
        import urllib.request
        from html.parser import HTMLParser

        class _P(HTMLParser):
            def __init__(self):
                super().__init__(); self.parts = []; self._skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ("script","style","nav","footer","header"): self._skip = True
            def handle_endtag(self, tag):
                if tag in ("script","style","nav","footer","header"): self._skip = False
            def handle_data(self, data):
                if not self._skip and data.strip(): self.parts.append(data.strip())

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        p = _P(); p.feed(html)
        return re.sub(r'\s+', ' ', " ".join(p.parts)).strip()
    except Exception as e:
        console.print(f"[red]Failed to fetch URL: {e}[/]")
        return ""


def clean_extracted_text(text: str) -> str:
    # Collapse spaces/tabs but preserve newlines so that strip_structure()
    # and the is_long_doc check (text.count('\n') > 30) work correctly on
    # multi-section documents (theses, reports).
    # r'\s+' was collapsing all whitespace including '\n', causing
    # strip_structure() to never fire and the document-level ensemble to
    # saturate on raw headers/TOC/reference lines.
    text = re.sub(r'[^\S\n]+', ' ', text)          # spaces/tabs → single space
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)    # non-ASCII → space (keep \n)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════
def print_banner():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]  AI Text Detector[/bold cyan]\n"
        "[dim]  DeBERTa-v3-large · XGBoost · Binoculars · Genre · Ensemble MLP[/dim]",
        border_style="cyan", padding=(1, 4),
    ))
    console.print()


def print_result_panel(result: dict, text_preview: str):
    label = result["label"]
    prob  = result["final_prob"]
    conf  = result["confidence"]
    color = "red" if label == "AI" else "green"
    emoji = "🤖" if label == "AI" else "👤"
    bar   = "█" * int(conf * 30) + "░" * (30 - int(conf * 30))

    # Genre display
    genre_info = result.get("genre_info", {})
    genre_name = genre_info.get("genre", "unknown")
    genre_form = result.get("genre_formality", 0.5)
    genre_line = (
        f"[dim]Genre           :[/dim]  {genre_name}  "
        f"[dim](formality: {genre_form*100:.1f}%)[/dim]\n"
    )

    console.print(Panel(
        f"[bold {color}]{emoji}  {label}  ({prob*100:.1f}% AI probability)"
        f"[/bold {color}]\n\n"
        f"[dim]Confidence[/dim]  [{color}]{bar}[/{color}] {conf*100:.1f}%\n\n"
        f"[dim]DeBERTa prob    :[/dim]  {result['deberta_prob']*100:.1f}%\n"
        f"[dim]XGBoost prob    :[/dim]  {result['xgb_prob']*100:.1f}%\n"
        f"[dim]Binoculars ratio:[/dim]  {result['bino_raw']:.3f}  "
        f"[dim](AI signal: {result['bino_ai_signal']*100:.1f}%)[/dim]\n"
        f"{genre_line}\n"
        f"[dim]Text preview:[/dim] [italic]{text_preview[:120]}...[/italic]",
        title="[bold]Detection Result[/bold]",
        border_style=color, padding=(1, 2),
    ))


def compute_sentence_stats(sent_results: list[dict]) -> dict:
    """
    Compute summary stats from sentence-level predictions.
    Returns counts and the AI sentence fraction for use in banners/warnings.
    """
    n_total = len(sent_results)
    n_ai    = sum(1 for s in sent_results if s["prob"] >= 0.50)
    n_hum   = sum(1 for s in sent_results if s["prob"] <= 0.35)
    n_unc   = n_total - n_ai - n_hum
    ai_frac = n_ai / n_total if n_total > 0 else 0.0
    return {
        "n_total": n_total,
        "n_ai":    n_ai,
        "n_hum":   n_hum,
        "n_unc":   n_unc,
        "ai_frac": ai_frac,
    }


# Long-document threshold: above this sentence count the heatmap is more reliable
_LONG_DOC_THRESHOLD = 50


def show_heatmap(models: DetectorModels, text: str, result: dict | None = None):
    console.print()
    console.print(Rule("[bold]Sentence Heatmap[/bold]", style="cyan"))
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[cyan]Analyzing sentences..."),
                  console=console, transient=True) as prog:
        prog.add_task("", total=None)
        sent_results = models.predict_sentences(text)

    if not sent_results:
        console.print("[yellow]Not enough sentences for heatmap.[/]")
        return

    output = Text()
    for sr in sent_results:
        p     = sr["prob"]
        style = "bold red" if p >= 0.50 else "bold green" if p <= 0.35 else "bold yellow"
        output.append("● ", style=style)
        output.append(sr["sentence"] + "  ", style=style)

    console.print(Panel(output, title="Heatmap", border_style="dim", padding=(1, 2)))

    legend = Table.grid(padding=(0, 3))
    legend.add_row(
        Text("● Human",     style="bold green"),
        Text("● Uncertain", style="bold yellow"),
        Text("● AI",        style="bold red"),
    )
    console.print(Align.center(legend))

    stats = compute_sentence_stats(sent_results)
    console.print()
    console.print(
        f"  [green]{stats['n_hum']} human[/green]  "
        f"[yellow]{stats['n_unc']} uncertain[/yellow]  "
        f"[red]{stats['n_ai']} AI[/red]  "
        f"[dim]out of {stats['n_total']} sentences[/dim]"
    )

    # ── Option 1: Long-document banner ───────────────────────────────────────
    if stats["n_total"] >= _LONG_DOC_THRESHOLD:
        ai_pct  = stats["ai_frac"] * 100
        hum_pct = (stats["n_hum"] / stats["n_total"]) * 100
        console.print()
        console.print(Panel(
            f"[bold yellow]⚠  Long document detected[/bold yellow] "
            f"[dim]({stats['n_total']} sentences)[/dim]\n\n"
            f"  Sentence-level analysis:  "
            f"[green]{stats['n_hum']} human ({hum_pct:.0f}%)[/green]  ·  "
            f"[red]{stats['n_ai']} AI ({ai_pct:.0f}%)[/red]  ·  "
            f"[yellow]{stats['n_unc']} uncertain[/yellow]\n\n"
            f"  [dim]For long structured documents (theses, reports, textbooks)\n"
            f"  the heatmap is more reliable than the ensemble headline score.\n"
            f"  Structural elements (headers, citations, TOC entries) can\n"
            f"  saturate the document-level models even when prose is human.[/dim]",
            border_style="yellow", padding=(0, 2),
        ))

    # ── Option 2: Score conflict warning ─────────────────────────────────────
    # Fires when sentence-level says mostly human but ensemble says mostly AI,
    # or vice versa.  Thresholds: ensemble ≥80% AI but <15% of sentences are AI,
    # or ensemble ≤20% AI but >85% of sentences are AI.
    if result is not None:
        ensemble_prob = result.get("final_prob", 0.5)
        sent_ai_frac  = stats["ai_frac"]
        conflict      = False
        conflict_msg  = ""

        if ensemble_prob >= 0.80 and sent_ai_frac < 0.15:
            conflict     = True
            conflict_msg = (
                f"  Ensemble score:       [red]{ensemble_prob*100:.1f}% AI[/red]  "
                f"[dim](document-level)[/dim]\n"
                f"  Sentence analysis:    [green]{sent_ai_frac*100:.1f}% of sentences flagged AI[/green]  "
                f"[dim](sentence-level)[/dim]\n\n"
                f"  [dim]The ensemble sees the document as highly AI-generated, but\n"
                f"  sentence analysis finds very little AI content. This often\n"
                f"  happens with formal/structured documents where headers, citations,\n"
                f"  and reference lists inflate the document-level score.\n"
                f"  [bold]Treat the sentence heatmap as the more reliable signal here.[/bold][/dim]"
            )
        elif ensemble_prob <= 0.20 and sent_ai_frac > 0.85:
            conflict     = True
            conflict_msg = (
                f"  Ensemble score:       [green]{ensemble_prob*100:.1f}% AI[/green]  "
                f"[dim](document-level)[/dim]\n"
                f"  Sentence analysis:    [red]{sent_ai_frac*100:.1f}% of sentences flagged AI[/red]  "
                f"[dim](sentence-level)[/dim]\n\n"
                f"  [dim]The ensemble rates this as likely human, but most individual\n"
                f"  sentences look AI-generated. Manual review is recommended.[/dim]"
            )

        if conflict:
            console.print()
            console.print(Panel(
                f"[bold red]⚠  Score conflict detected[/bold red]\n\n"
                f"{conflict_msg}",
                border_style="red", padding=(0, 2),
            ))

    console.print()


def show_explain(result: dict):
    console.print()
    console.print(Rule("[bold]Detection Explanation[/bold]", style="cyan"))
    console.print()

    genre_info = result.get("genre_info", {})
    genre_name = genre_info.get("genre", "unknown")

    signals = {
        "DeBERTa (semantic)":      result["deberta_prob"],
        "XGBoost (stylometric)":   result["xgb_prob"],
        "Binoculars (perplexity)": result["bino_ai_signal"],
        "Genre formality":         result.get("genre_formality", 0.5),
        "Ensemble final":          result["final_prob"],
    }

    bar_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    bar_table.add_column("Signal",  style="dim", width=28)
    bar_table.add_column("Bar",     width=34)
    bar_table.add_column("Score",   style="bold", width=8, justify="right")
    bar_table.add_column("Verdict", width=10)

    for name, score in signals.items():
        filled  = int(score * 32)
        color   = "red" if score >= 0.65 else "green" if score <= 0.35 else "yellow"
        bar     = f"[{color}]{'█'*filled}[/{color}][dim]{'░'*(32-filled)}[/dim]"
        verdict = (f"[red]AI[/red]" if score >= 0.65
                   else f"[green]Human[/green]" if score <= 0.35
                   else f"[yellow]Unsure[/yellow]")
        bar_table.add_row(name, bar, f"{score*100:.1f}%", verdict)

    console.print(bar_table)

    from features import FeatureExtractor
    feat_names = FeatureExtractor.FEATURE_NAMES
    feats      = result["features"]

    console.print()
    console.print("[bold dim]Key stylometric signals:[/bold dim]")
    mt = Table(box=box.SIMPLE_HEAVY, padding=(0, 2))
    mt.add_column("Feature", style="dim")
    mt.add_column("Value",   justify="right")
    mt.add_column("Meaning", style="dim italic")

    for fname, meaning in [
        ("burstiness",         "Higher = more human-like variance"),
        ("unique_word_ratio",  "Higher = richer vocabulary"),
        ("ai_filler_density",  "Higher = more AI-like phrases"),
        ("hedge_word_ratio",   "Higher = more AI hedging"),
        ("avg_sent_len_words", "AI tends to write longer sentences"),
        ("unigram_entropy",    "Higher = more diverse word choice"),
    ]:
        if fname in feat_names:
            mt.add_row(fname, f"{feats[feat_names.index(fname)]:.3f}", meaning)

    console.print(mt)
    console.print()


def save_pdf_report(result: dict, text: str, output_path: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table as RLTable, TableStyle, HRFlowable,
        )
        import datetime
    except ImportError:
        console.print("[red]reportlab not installed: pip install reportlab[/]")
        return False

    doc    = SimpleDocTemplate(str(output_path), pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm,  bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph(
        "AI Text Detection Report",
        ParagraphStyle("t", parent=styles["Title"], fontSize=22, spaceAfter=6,
                       textColor=colors.HexColor("#1a1a2e"))
    ))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.4*cm))

    label   = result["label"]
    prob    = result["final_prob"]
    v_color = (colors.HexColor("#cc0000") if label == "AI"
               else colors.HexColor("#006600"))
    story.append(Paragraph(
        f"Verdict: {label}  ({prob*100:.1f}% AI probability)",
        ParagraphStyle("v", parent=styles["Heading1"], fontSize=18,
                       textColor=v_color, spaceAfter=4)
    ))
    story.append(Paragraph(
        f"Confidence: {result['confidence']*100:.1f}%", styles["Normal"]
    ))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Signal Breakdown", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))

    tdata = [["Signal", "Score", "Verdict"]]
    genre_info = result.get("genre_info", {})
    genre_name = genre_info.get("genre", "unknown")

    for name, score in [
        ("DeBERTa (semantic)",      result["deberta_prob"]),
        ("XGBoost (stylometric)",   result["xgb_prob"]),
        ("Binoculars (perplexity)", result["bino_ai_signal"]),
        (f"Genre ({genre_name})",   result.get("genre_formality", 0.5)),
        ("Ensemble final",          result["final_prob"]),
    ]:
        v = "AI" if score >= 0.65 else "Human" if score <= 0.35 else "Uncertain"
        tdata.append([name, f"{score*100:.1f}%", v])

    t = RLTable(tdata, colWidths=[9*cm, 3*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    preview = text[:800] + ("..." if len(text) > 800 else "")
    story.append(Paragraph("Analyzed Text (preview)", styles["Heading2"]))
    story.append(Paragraph(
        preview.replace("\n", "<br/>"),
        ParagraphStyle("p", parent=styles["Normal"], fontSize=9,
                       textColor=colors.HexColor("#444444"),
                       backColor=colors.HexColor("#f9f9f9"),
                       leftIndent=10, rightIndent=10,
                       spaceBefore=4, spaceAfter=4)
    ))
    doc.build(story)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# File browser
# ══════════════════════════════════════════════════════════════════════════════
def _open_file_dialog(title: str, multiple: bool = False):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        filetypes = [
            ("Supported files", "*.pdf *.docx *.txt"),
            ("PDF files",       "*.pdf"),
            ("Word documents",  "*.docx"),
            ("Text files",      "*.txt"),
            ("All files",       "*.*"),
        ]
        if multiple:
            paths  = filedialog.askopenfilenames(title=title, filetypes=filetypes)
            result = [Path(p) for p in paths]
        else:
            path   = filedialog.askopenfilename(title=title, filetypes=filetypes)
            result = [Path(path)] if path else []
        root.destroy()
        return result
    except Exception as e:
        console.print(f"[yellow]File dialog unavailable ({e}). "
                      f"Falling back to manual entry.[/yellow]")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Post-analysis menu
# ══════════════════════════════════════════════════════════════════════════════
def post_analysis_menu(models: DetectorModels, result: dict, text: str):
    while True:
        console.print()
        console.print(Rule("[dim]Analysis options[/dim]", style="dim"))
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column(style="bold cyan", width=4)
        table.add_column()
        table.add_row("1", "[green]Heatmap[/green]  — highlight each sentence")
        table.add_row("2", "[blue]Explain[/blue]   — bar chart + all signal breakdown")
        table.add_row("3", "[magenta]Report[/magenta]   — save as PDF")
        table.add_row("4", "[dim]Back[/dim]      — return to main menu")
        console.print(table)

        choice = Prompt.ask(
            "\n[bold cyan]Choose option[/bold cyan]",
            choices=["1", "2", "3", "4"],
            default="4",
        )

        if choice == "1":
            show_heatmap(models, text, result)
        elif choice == "2":
            show_explain(result)
        elif choice == "3":
            fname    = Prompt.ask("[cyan]Save PDF as[/cyan]",
                                  default="detection_report.pdf")
            out_path = Path(fname)
            with Progress(SpinnerColumn(), TextColumn("[cyan]Generating PDF..."),
                          console=console, transient=True) as prog:
                prog.add_task("", total=None)
                ok = save_pdf_report(result, text, out_path)
            if ok:
                console.print(f"[green]Report saved:[/green] {out_path.resolve()}")
            else:
                console.print("[red]Failed to save PDF.[/red]")
        elif choice == "4":
            break


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3: Analysis runner — structural stripping before document-level scoring
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(models: DetectorModels, text: str, label: str = ""):
    """
    Entry point for a single analysis.

    Two-track approach:
      - Document-level score: uses strip_structure() on multi-paragraph
        documents to remove TOC entries, headers, reference lines, and
        other structural elements that saturate DeBERTa on formal texts.
      - Heatmap: always receives the original raw text so sentence
        colouring is never affected by structural stripping.
    """
    text = clean_extracted_text(text)
    if len(text.split()) < 20:
        console.print("[red]Text too short — need at least 20 words.[/]")
        return

    # Determine whether structural stripping is warranted.
    # Only strip when the document is long enough to contain sections
    # (>30 newlines is a reasonable proxy for a multi-section document).
    is_long_doc = text.count('\n') > 30
    if is_long_doc:
        text_for_doc = strip_structure(text)
        # Safety net: if stripping removes too much (< 100 words remain),
        # fall back to the original text to avoid scoring near-empty input.
        if len(text_for_doc.split()) < 100:
            text_for_doc = text
    else:
        text_for_doc = text

    with Progress(SpinnerColumn(),
                  TextColumn("[bold cyan]Running detection pipeline..."),
                  console=console, transient=True) as prog:
        prog.add_task("", total=None)
        result = models.predict(text_for_doc)   # scored on clean prose

    print_result_panel(result, text)            # preview shows original text
    post_analysis_menu(models, result, text)    # heatmap uses original text


# ══════════════════════════════════════════════════════════════════════════════
# Input handlers
# ══════════════════════════════════════════════════════════════════════════════
def handle_paste(models: DetectorModels):
    console.print()
    console.print(Panel(
        "[dim]Paste your text below.\n"
        "When done, enter [bold]END[/bold] on a new line and press Enter.[/dim]",
        border_style="cyan", padding=(1, 2),
    ))
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    text = "\n".join(lines)
    if text.strip():
        run_analysis(models, text, "Pasted text")
    else:
        console.print("[yellow]No text entered.[/yellow]")


def handle_single_file(models: DetectorModels):
    console.print()
    paths = _open_file_dialog("Select a file to analyze")
    if paths is None:
        path_str = Prompt.ask(
            "[cyan]Enter full file path[/cyan] [dim](PDF / DOCX / TXT)[/dim]"
        )
        paths = [Path(path_str.strip().strip('"'))]
    if not paths:
        console.print("[yellow]No file selected.[/yellow]")
        return
    path = paths[0]
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return
    console.print(f"[dim]Selected:[/dim] [cyan]{path.name}[/cyan]")
    with Progress(SpinnerColumn(),
                  TextColumn(f"[cyan]Reading {path.name}..."),
                  console=console, transient=True) as prog:
        prog.add_task("", total=None)
        text = extract_text_from_file(path)
    if text.strip():
        run_analysis(models, text, path.name)
    else:
        console.print("[red]Could not extract text from file.[/red]")


def handle_batch(models: DetectorModels):
    console.print()
    console.print(Panel(
        "[dim]A file picker will open — hold [bold]Ctrl[/bold] "
        "to select multiple files.[/dim]",
        border_style="cyan", padding=(1, 2),
    ))
    paths = _open_file_dialog(
        "Select files to analyze (Ctrl+click for multiple)", multiple=True
    )
    if paths is None:
        console.print(Panel(
            "[dim]Enter file paths one per line.\n"
            "When done, enter [bold]END[/bold] on a new line.[/dim]",
            border_style="cyan", padding=(1, 2),
        ))
        paths = []
        while True:
            try:
                line = input().strip().strip('"')
            except EOFError:
                break
            if line.upper() == "END":
                break
            if line:
                paths.append(Path(line))
    if not paths:
        console.print("[yellow]No files selected.[/yellow]")
        return
    console.print(f"\n[cyan]Processing {len(paths)} file(s) ...[/cyan]\n")
    for i, path in enumerate(paths, 1):
        console.print(Rule(f"[bold]File {i}/{len(paths)}: {path.name}[/bold]",
                           style="cyan"))
        if not path.exists():
            console.print(f"[red]Not found: {path}[/red]")
            continue
        text = extract_text_from_file(path)
        if text.strip():
            run_analysis(models, text, path.name)
        else:
            console.print(f"[red]Could not read: {path.name}[/red]")


def handle_url(models: DetectorModels):
    console.print()
    url = Prompt.ask("[cyan]Enter URL[/cyan]").strip()
    if not url.startswith("http"):
        url = "https://" + url
    with Progress(SpinnerColumn(),
                  TextColumn(f"[cyan]Fetching {url[:60]}..."),
                  console=console, transient=True) as prog:
        prog.add_task("", total=None)
        text = extract_text_from_url(url)
    if text.strip():
        run_analysis(models, text, f"URL: {url[:50]}")
    else:
        console.print("[red]Could not extract text from URL.[/red]")


# ══════════════════════════════════════════════════════════════════════════════
# Main menu
# ══════════════════════════════════════════════════════════════════════════════
def main_menu(models: DetectorModels):
    while True:
        console.print()
        console.print(Rule(style="dim cyan"))
        menu = Table(box=box.SIMPLE, show_header=False,
                     padding=(0, 2), border_style="dim")
        menu.add_column(style="bold cyan", width=4)
        menu.add_column(width=56)
        menu.add_row("1", "[bold]Paste text[/bold]   — type or paste text directly")
        menu.add_row("2", "[bold]Single file[/bold]  — analyze a PDF, DOCX, or TXT")
        menu.add_row("3", "[bold]Batch files[/bold]  — analyze multiple files at once")
        menu.add_row("4", "[bold]URL[/bold]          — fetch and analyze a webpage")
        menu.add_row("5", "[dim]Exit[/dim]")
        console.print(menu)

        choice = Prompt.ask(
            "\n[bold cyan]Choose option[/bold cyan]",
            choices=["1", "2", "3", "4", "5"],
            default="5",
        )

        if   choice == "1": handle_paste(models)
        elif choice == "2": handle_single_file(models)
        elif choice == "3": handle_batch(models)
        elif choice == "4": handle_url(models)
        elif choice == "5":
            console.print(Panel("[bold cyan]Goodbye![/bold cyan]",
                                border_style="cyan", padding=(1, 4)))
            break


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print_banner()
    console.print("[bold cyan]Loading models...[/bold cyan]")
    console.print(
        "[dim]First run downloads gpt2-xl (~6GB). Subsequent runs load from cache.[/dim]\n"
    )

    models = DetectorModels()
    try:
        models.load()
    except Exception as e:
        console.print(f"[red]Failed to load models: {e}[/red]")
        console.print("[dim]Make sure you have run all training scripts:[/dim]")
        console.print("  [cyan]python src/train_deberta.py[/cyan]")
        console.print("  [cyan]python src/train_xgboost.py  (twice)[/cyan]")
        console.print("  [cyan]python src/train_ensemble.py[/cyan]")
        return

    console.print("\n[bold green]All models loaded.[/bold green]\n")
    main_menu(models)


if __name__ == "__main__":
    main()