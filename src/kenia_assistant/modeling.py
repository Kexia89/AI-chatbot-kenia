from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_intent_pipeline(c: float = 8.0, char_weight: float = 1.0) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    strip_accents="unicode",
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": char_weight},
    )
    classifier = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=2500,
        solver="lbfgs",
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])
