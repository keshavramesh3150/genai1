from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional
import random
import re

import numpy as np
import spacy


def _load_spacy_model(preferred: str = "en_core_web_lg") -> "spacy.language.Language":
    """
    Load a spaCy model, preferring the large vectors model.
    Falls back to medium/small if the large model isn't installed.
    """
    tried = []
    for name in (preferred, "en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(name)
        except Exception as e:
            tried.append((name, str(e)))
    hints = "\n".join([f"- {n}: {msg}" for n, msg in tried])
    raise RuntimeError(
        "Could not load a spaCy English model. "
        "Install one of: en_core_web_lg / en_core_web_md / en_core_web_sm\n"
        "Example: uv run python -m spacy download en_core_web_lg\n\n"
        f"Tried:\n{hints}"
    )


class BigramModel:
    """
    Bigram language model with spaCy word-embedding utilities.

    - Build bigram counts & smoothed probabilities from a list of sentences.
    - Generate text by sampling next words from the learned bigram distribution.
    - Backoff: if no outgoing bigrams for a word, map to the nearest semantic
      neighbor (via embeddings) that *does* have followers.
    - Exposes get_embedding() for the embedding API.
    """

    def __init__(
        self,
        corpus: List[str],
        lowercase: bool = True,
        seed: Optional[int] = 42,
        laplace_alpha: float = 1.0,
    ):
        if seed is not None:
            random.seed(seed)

        self.lowercase = lowercase
        self.alpha = laplace_alpha

        # Load spaCy model (with vectors if available)
        self.nlp = _load_spacy_model()

        # Tokenize into sentence-level token lists, add <s> and </s>
        self.sent_tokens: List[List[str]] = [
            ["<s>"] + self._tokenize(line) + ["</s>"] for line in corpus
        ]

        # Build bigram & unigram counts
        self.bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.unigram_counts: Counter = Counter()

        for sent in self.sent_tokens:
            for i in range(len(sent) - 1):
                w1, w2 = sent[i], sent[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
            self.unigram_counts[sent[-1]] += 1

        # Words that appear as current tokens
        self.vocab: List[str] = list(self.bigram_counts.keys())

        # Cache embeddings for alpha-only tokens (skip specials)
        self._emb_cache: Dict[str, np.ndarray] = {}
        for w in self.vocab:
            if w not in ("<s>", "</s>") and re.match(r"^[A-Za-z]+$", w):
                self._emb_cache[w] = self.nlp(w).vector

    # ----------------------------
    # Public API
    # ----------------------------
    def generate_text(self, start_word: str, length: int = 10) -> str:
        """
        Generate text of `length` words starting from `start_word`.
        Uses Laplace-smoothed bigram sampling with embedding backoff.
        """
        current = self._normalize(start_word)
        if current not in self.bigram_counts:
            mapped = self._closest_in_vocab(current)
            current = mapped if mapped is not None else "<s>"

        output: List[str] = []
        for _ in range(max(0, length)):
            nxt = self._sample_next(current)
            if nxt == "</s>":
                current = "<s>"
                continue
            if nxt in ("<s>", "</s>"):
                continue
            output.append(nxt)
            current = nxt

        return " ".join(output) if output else current

    def get_embedding(self, word: str) -> List[float]:
        """
        Return the spaCy embedding for a single word as a Python list.
        """
        vec = self.nlp(self._normalize(word)).vector
        return vec.astype(float).tolist()

    # ----------------------------
    # Internals
    # ----------------------------
    def _normalize(self, text: str) -> str:
        return text.lower().strip() if self.lowercase else text.strip()

    def _tokenize(self, text: str) -> List[str]:
        """
        spaCy tokenization; keep alphabetic tokens; drop numbers/punct/space.
        """
        doc = self.nlp(text)
        toks = []
        for t in doc:
            if t.is_space or t.is_punct or t.like_num:
                continue
            toks.append(self._normalize(t.text))
        return toks

    def _sample_next(self, current: str) -> str:
        followers = self.bigram_counts.get(current, None)
        if followers and len(followers) > 0:
            return self._draw_smoothed(followers)

        neighbor = self._closest_in_vocab(current)
        if neighbor is not None and len(self.bigram_counts[neighbor]) > 0:
            return self._draw_smoothed(self.bigram_counts[neighbor])

        # Fallback: most common unigram (not special tokens)
        for candidate, _ in self.unigram_counts.most_common():
            if candidate not in ("<s>", "</s>"):
                return candidate
        return "</s>"

    def _draw_smoothed(self, counter: Counter) -> str:
        items = list(counter.items())
        words = [w for w, _ in items]
        counts = np.array([c for _, c in items], dtype=float)
        counts += self.alpha  # Laplace smoothing
        probs = counts / counts.sum()
        return str(np.random.choice(words, p=probs))

    def _closest_in_vocab(self, query: str) -> Optional[str]:
        """
        Find most embedding-similar vocab word that has outgoing bigrams.
        """
        q_vec = self.nlp(self._normalize(query)).vector
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0.0:
            return None

        best_word = None
        best_sim = -1.0

        for w, v in self._emb_cache.items():
            if len(self.bigram_counts.get(w, {})) == 0:
                continue
            denom = q_norm * np.linalg.norm(v)
            if denom == 0.0:
                continue
            sim = float(np.dot(q_vec, v) / denom)
            if sim > best_sim:
                best_sim = sim
                best_word = w

        return best_word