from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

log = logging.getLogger("echo.embed")


class LSAEncoder:
    """Always-on lexical-semantic vectors. Fit once at ingest, transform in <2ms."""

    def __init__(self, n_features: int = 20000, n_components: int = 256) -> None:
        self.n_features = n_features
        self.n_components = n_components
        self.tfidf: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.dim = n_components

    def fit(self, texts: list[str]) -> None:
        self.tfidf = TfidfVectorizer(
            max_features=self.n_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            token_pattern=r"(?u)[A-Za-z0-9]{2,}|[\u0900-\u097F]{2,}",
        )
        sparse = self.tfidf.fit_transform(texts)
        n_comp = min(self.n_components, max(2, sparse.shape[0] - 1), sparse.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=7)
        self.svd.fit(sparse)
        self.dim = n_comp
        log.info("LSA fitted dim=%s vocab=%s", self.dim, len(self.tfidf.vocabulary_))

    def encode(self, texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
        assert self.tfidf is not None and self.svd is not None
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            vec = self.svd.transform(self.tfidf.transform(chunk))
            parts.append(sk_normalize(vec))
        return np.vstack(parts).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
