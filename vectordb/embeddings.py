"""
embeddings.py — text to vector, at a scale where the naive dense
approach stops working.

Same idea as classical TF-IDF+LSA, rebuilt for 5,000+ documents:
  - TF-IDF matrix is built and kept SPARSE (scipy.sparse) — a
    5,000-doc x large-vocab dense matrix wastes enormous memory for what
    is >99% zeros.
  - SVD is computed with scipy.sparse.linalg.svds, a truncated solver
    that only computes the top-k singular vectors instead of a full
    O(n^2 m) dense decomposition, which is the thing that doesn't scale.

Scope note: this file turns TEXT INTO VECTORS. The actual vector
DATABASE — exact index, approximate index, insert/search/delete — is in
vectordb/core.py and uses nothing but numpy, which is what "no ANN
library" is actually about. TF-IDF and SVD are classical linear-algebra
feature extraction, not a vector search engine.
"""

import re
import math
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "was", "were", "are", "be",
    "to", "of", "in", "on", "at", "for", "it", "this", "that", "my", "i",
    "you", "your", "me", "with", "as", "just", "so", "if",
}


def tokenize(text: str, use_bigrams: bool = True):
    words = [t.lower() for t in _TOKEN_RE.findall(text)]
    tokens = list(words)
    if use_bigrams:
        content = [w for w in words if w not in _STOPWORDS]
        tokens += [content[i] + "_" + content[i + 1] for i in range(len(content) - 1)]
    return tokens


class TfidfEmbedder:
    def __init__(self, n_components: int = 96):
        self.n_components = n_components
        self.vocab = {}
        self.idf = None
        self.components = None  # (n_vocab, k)

    def _doc_freqs(self, docs):
        df = Counter()
        for doc in docs:
            for t in set(tokenize(doc)):
                df[t] += 1
        return df

    def _build_tfidf(self, docs):
        rows, cols, vals = [], [], []
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            total = sum(counts.values()) or 1
            for tok, cnt in counts.items():
                idx = self.vocab.get(tok)
                if idx is None:
                    continue
                tf = cnt / total
                rows.append(i)
                cols.append(idx)
                vals.append(tf * self.idf[idx])
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)))

    def fit(self, docs: list) -> np.ndarray:
        n_docs = len(docs)
        doc_freq = self._doc_freqs(docs)
        self.vocab = {t: i for i, t in enumerate(sorted(doc_freq.keys()))}

        self.idf = np.zeros(len(self.vocab), dtype=np.float64)
        for tok, idx in self.vocab.items():
            df = doc_freq[tok]
            self.idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        # clip extreme idf so a single rare shared word can't single-handedly
        # dominate cosine similarity between two otherwise-unrelated documents
        cap = np.percentile(self.idf, 85)
        self.idf = np.minimum(self.idf, cap)

        tfidf = self._build_tfidf(docs)
        k = min(self.n_components, min(tfidf.shape) - 1)
        u, s, vt = svds(tfidf.astype(np.float64), k=k)
        order = np.argsort(-s)          # svds returns ascending singular values
        vt = vt[order, :]
        self.components = vt.T          # (n_vocab, k)

        dense = tfidf @ self.components
        return np.asarray(dense, dtype=np.float32)

    def transform(self, docs: list) -> np.ndarray:
        tfidf = self._build_tfidf(docs)
        dense = tfidf @ self.components
        return np.asarray(dense, dtype=np.float32)
