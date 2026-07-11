import re
import math
import logging
import warnings
from collections import Counter
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Optional heavy deps (graceful fallback if not installed) ───────────────────
try:
    import textstat
    TEXTSTAT_OK = True
except ImportError:
    TEXTSTAT_OK = False
    log.warning("textstat not installed -- readability features will be zeros. "
                "pip install textstat")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    # Download quietly if not present
    import ssl
    try:
        _create_default_https_context = ssl._create_default_https_context
    except AttributeError:
        pass
    nltk.download("punkt",        quiet=True)
    nltk.download("punkt_tab",    quiet=True)
    nltk.download("stopwords",    quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
    NLTK_OK = True
except Exception:
    NLTK_OK = False
    STOPWORDS = set()
    log.warning("nltk not fully available -- some features will be approximate.")

# ── Constants ──────────────────────────────────────────────────────────────────
# Common AI "tells" — phrases that appear disproportionately in LLM output
AI_FILLER_PHRASES = [
    "it is important to note",
    "it is worth noting",
    "in conclusion",
    "in summary",
    "to summarize",
    "overall,",
    "furthermore,",
    "moreover,",
    "additionally,",
    "it is essential",
    "plays a crucial role",
    "it is crucial",
    "delve into",
    "dive into",
    "in today's world",
    "in the modern world",
    "as an ai",
    "as a language model",
    "i cannot",
    "i'm unable to",
    "certainly!",
    "absolutely!",
    "of course!",
    "great question",
    "that's a great",
]

# Hedge words — AI overuses these
HEDGE_WORDS = {
    "perhaps", "possibly", "probably", "generally", "typically",
    "usually", "often", "sometimes", "certain", "various",
    "numerous", "significant", "important", "key", "crucial",
    "essential", "effective", "efficient", "relevant", "appropriate",
}

# Function words — ratio should differ between human/AI writing
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "and", "or", "but", "if", "then", "because", "so", "yet",
    "for", "nor", "not", "in", "on", "at", "to", "of", "by",
    "with", "about", "against", "between", "through", "during",
    "this", "that", "these", "those", "it", "its",
}

# NEW: Personal pronouns - humans use these more
PERSONAL_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
}

# NEW: Transition words - AI overuses these
TRANSITION_WORDS = {
    "first", "second", "third", "finally", "lastly",
    "additionally", "furthermore", "moreover", "however",
    "nevertheless", "in addition", "on the other hand",
    "therefore", "thus", "consequently", "hence",
    "meanwhile", "subsequently", "accordingly",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_sentences(text: str) -> list[str]:
    """Split text into sentences, fallback to period-split if nltk unavailable."""
    if NLTK_OK:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def _safe_words(text: str) -> list[str]:
    """Tokenize text into words, fallback to regex split."""
    if NLTK_OK:
        try:
            return word_tokenize(text.lower())
        except Exception:
            pass
    return re.findall(r'\b[a-z]+\b', text.lower())

def _entropy(counter: Counter, total: int) -> float:
    """Shannon entropy of a frequency distribution."""
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counter.values() if c > 0
    )

# ── Feature Extractor ──────────────────────────────────────────────────────────
class FeatureExtractor:
    """
    Extracts 39 stylometric features from a text string (was 36, added 3 new ones).
    All features are normalized to [0, 1] or z-score range
    so XGBoost doesn't need additional scaling.
    """
    FEATURE_NAMES = [
        # --- sentence stats (6) ---
        "sent_count",
        "avg_sent_len_words",
        "std_sent_len_words",
        "burstiness",               # std / mean sentence perplexity proxy
        "max_sent_len",
        "min_sent_len",
        
        # --- word stats (6) ---
        "word_count",
        "avg_word_len_chars",
        "std_word_len_chars",
        "unique_word_ratio",        # type-token ratio (TTR)
        "hapax_ratio",              # words appearing exactly once / total
        "long_word_ratio",          # words > 6 chars / total
        
        # --- vocabulary richness (3) ---
        "vocab_size",
        "function_word_ratio",
        "hedge_word_ratio",
        
        # --- punctuation (6) ---
        "comma_per_word",
        "period_per_word",
        "exclamation_per_word",
        "question_per_word",
        "semicolon_per_word",
        "colon_per_word",
        
        # --- readability (3) ---
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        
        # --- entropy (3) ---
        "unigram_entropy",
        "bigram_entropy",
        "char_bigram_entropy",
        
        # --- AI-tell signals (5) ---
        "ai_filler_count",          # count of known AI filler phrases
        "ai_filler_density",        # filler count / sentence count
        "list_starter_ratio",       # sentences starting with bullet signals
        "uppercase_ratio",          # ALL CAPS words / total words
        "digit_ratio",              # words containing digits / total words
        
        # --- structural (4) ---
        "avg_paragraph_len",
        "paragraph_count",
        "sent_len_entropy",         # entropy of sentence length distribution
        "stopword_ratio",
        
        # --- NEW: High-value features (3) ---
        "personal_pronoun_ratio",   # humans use I/me/my more
        "transition_word_density",  # AI overuses transitions
        "repeat_phrase_ratio",      # AI repeats phrases more
    ]
    
    def __init__(self):
        assert len(self.FEATURE_NAMES) == 39, \
            f"Expected 39 features, got {len(self.FEATURE_NAMES)}"
    
    def extract(self, text: str) -> np.ndarray:
        """
        Extract all features from a single text.
        Returns np.ndarray of shape (39,).
        Safe — never raises, returns zeros on failure.
        """
        try:
            return self._extract_safe(text)
        except Exception as e:
            log.debug("Feature extraction failed for text: %s", str(e))
            return np.zeros(39, dtype=np.float32)
    
    def _extract_safe(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(39, dtype=np.float32)
        
        text_lower = text.lower()
        words      = _safe_words(text)
        sentences  = _safe_sentences(text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        n_words = max(len(words), 1)
        n_sents = max(len(sentences), 1)
        n_paras = max(len(paragraphs), 1)
        
        # ── Sentence stats ──────────────────────────────────────────────────────
        sent_lens = [len(_safe_words(s)) for s in sentences]
        avg_sent  = np.mean(sent_lens)
        std_sent  = np.std(sent_lens)
        burstiness = std_sent / max(avg_sent, 1)  # high = human-like variance
        
        # ── Word stats ──────────────────────────────────────────────────────────
        word_lens    = [len(w) for w in words]
        word_counter = Counter(words)
        hapax        = sum(1 for c in word_counter.values() if c == 1)
        long_words   = sum(1 for w in words if len(w) > 6)
        
        # ── Vocabulary ──────────────────────────────────────────────────────────
        unique_ratio   = len(word_counter) / n_words
        hapax_ratio    = hapax / n_words
        long_word_ratio = long_words / n_words
        func_words     = sum(word_counter.get(w, 0) for w in FUNCTION_WORDS)
        hedge_words    = sum(word_counter.get(w, 0) for w in HEDGE_WORDS)
        
        # ── Punctuation ─────────────────────────────────────────────────────────
        def _punct_density(char):
            return text.count(char) / n_words
        
        # ── Readability ─────────────────────────────────────────────────────────
        if TEXTSTAT_OK:
            try:
                flesch_ease  = textstat.flesch_reading_ease(text)
                flesch_grade = textstat.flesch_kincaid_grade(text)
                fog          = textstat.gunning_fog(text)
                # Clamp to reasonable range
                flesch_ease  = max(-100, min(120, flesch_ease))
                flesch_grade = max(0,    min(20,  flesch_grade))
                fog          = max(0,    min(30,  fog))
            except Exception:
                flesch_ease = flesch_grade = fog = 0.0
        else:
            flesch_ease = flesch_grade = fog = 0.0
        
        # ── Entropy ─────────────────────────────────────────────────────────────
        unigram_ent = _entropy(word_counter, n_words)
        
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_ent = _entropy(Counter(bigrams), max(len(bigrams), 1))
        
        chars = list(text_lower.replace("  ", " "))
        char_bigrams = list(zip(chars[:-1], chars[1:]))
        char_bigram_ent = _entropy(Counter(char_bigrams), max(len(char_bigrams), 1))
        
        # ── AI-tell signals ─────────────────────────────────────────────────────
        ai_filler_count = sum(
            text_lower.count(phrase) for phrase in AI_FILLER_PHRASES
        )
        ai_filler_density = ai_filler_count / n_sents
        
        # Sentences starting with transition markers
        list_starters = sum(
            1 for s in sentences
            if re.match(
                r'^\s*(first|second|third|finally|lastly|additionally|'
                r'furthermore|moreover|however|nevertheless|in addition|'
                r'on the other hand)',
                s.lower()
            )
        )
        list_starter_ratio = list_starters / n_sents
        
        all_words_raw = text.split()
        uppercase_count = sum(1 for w in all_words_raw
                              if w.isupper() and len(w) > 1)
        digit_count = sum(1 for w in words if any(c.isdigit() for c in w))
        
        # ── Structural ──────────────────────────────────────────────────────────
        avg_para_len = np.mean([len(_safe_words(p)) for p in paragraphs])
        
        sent_len_counter = Counter(sent_lens)
        sent_len_ent = _entropy(sent_len_counter, n_sents)
        
        stopword_count = sum(
            1 for w in words if w in STOPWORDS
        ) if STOPWORDS else 0
        
        # ── NEW: High-value features ────────────────────────────────────────────
        
        # 1. Personal pronoun ratio - humans use "I", "me", "my" more
        personal_pronoun_count = sum(
            word_counter.get(w, 0) for w in PERSONAL_PRONOUNS
        )
        personal_pronoun_ratio = personal_pronoun_count / n_words
        
        # 2. Transition word density - AI overuses transitions
        transition_word_count = sum(
            word_counter.get(w, 0) for w in TRANSITION_WORDS
        )
        transition_word_density = transition_word_count / n_words
        
        # 3. Repeat phrase ratio - AI tends to repeat phrases
        # Count bigrams that appear more than once
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        repeat_phrase_ratio = repeated_bigrams / max(len(bigrams), 1)
        
        # ── Assemble feature vector ─────────────────────────────────────────────
        features = np.array([
            # sentence stats
            n_sents,
            avg_sent,
            std_sent,
            burstiness,
            max(sent_lens),
            min(sent_lens),
            # word stats
            n_words,
            np.mean(word_lens) if word_lens else 0,
            np.std(word_lens)  if word_lens else 0,
            unique_ratio,
            hapax_ratio,
            long_word_ratio,
            # vocabulary
            len(word_counter),
            func_words / n_words,
            hedge_words / n_words,
            # punctuation
            _punct_density(","),
            _punct_density("."),
            _punct_density("!"),
            _punct_density("?"),
            _punct_density(";"),
            _punct_density(":"),
            # readability
            flesch_ease,
            flesch_grade,
            fog,
            # entropy
            unigram_ent,
            bigram_ent,
            char_bigram_ent,
            # AI-tell
            ai_filler_count,
            ai_filler_density,
            list_starter_ratio,
            uppercase_count / n_words,
            digit_count / n_words,
            # structural
            avg_para_len,
            n_paras,
            sent_len_ent,
            stopword_count / n_words,
            # NEW features
            personal_pronoun_ratio,
            transition_word_density,
            repeat_phrase_ratio,
        ], dtype=np.float32)
        
        # Replace any NaN/Inf that slipped through
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def extract_dataframe(
        self,
        texts: list[str],
        labels: Optional[list[int]] = None,
        batch_size: int = 1000,
        desc: str = "Extracting features",
    ) -> pd.DataFrame:
        """
        Extract features for a list of texts.
        Returns a DataFrame with columns = FEATURE_NAMES (+ 'label' if provided).
        Processes in batches and shows a progress bar.
        """
        all_features = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i : i + batch_size]
            for text in batch:
                all_features.append(self.extract(text))
        
        df = pd.DataFrame(all_features, columns=self.FEATURE_NAMES)
        
        if labels is not None:
            df["label"] = labels
        
        return df

# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    extractor = FeatureExtractor()
    
    human_text = (
        "I went to the store yesterday and honestly it was a mess. "
        "The shelves were half empty — no bread anywhere. "
        "I grabbed some milk and left. Not worth the trip."
    )
    
    ai_text = (
        "It is important to note that grocery shopping can be a complex activity "
        "that requires careful planning and consideration. Furthermore, various "
        "factors such as store layout, product availability, and customer behavior "
        "play a crucial role in determining the overall shopping experience. "
        "In conclusion, it is essential to approach this task with a clear strategy."
    )
    
    h_feat = extractor.extract(human_text)
    a_feat = extractor.extract(ai_text)
    
    print("\n--- Feature comparison ---")
    print(f"{'Feature': <28} {'Human': >10} {'AI': >10}")
    print("-" * 50)
    for name, hv, av in zip(FeatureExtractor.FEATURE_NAMES, h_feat, a_feat):
        if abs(hv - av) > 0.01:   # only show features that differ
            print(f"{name: <28} {hv: >10.3f} {av: >10.3f}")
    
    print(f"\nFeature vector shape: {h_feat.shape}")
    print("features.py OK")