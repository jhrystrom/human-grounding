"""Surface-form lexical baselines: TF-IDF char-ngrams and binary Jaccard.

Both expose the same ``Embedder`` interface as the neural models so they
flow through ``get_statement_embeddings`` and the downstream triplet-alpha
and clustering pipelines unchanged. Each call fits the vectoriser on
the supplied corpus (per-dataset; ≤ ~120 statements), so the output
dimension is fixed within a call but corpus-dependent across calls.
"""

from collections.abc import Callable
from functools import partial

from joblib import Memory

from human_grounding.data_models import Embedder
from human_grounding.directories import CACHE_DIR

MODELS = ("tfidf-char35", "jaccard-binary")
AVAILABLE_MODELS = set(MODELS)
memory = Memory(CACHE_DIR, verbose=0)


@memory.cache
def _tfidf_char_encode(text: list[str], model_name: str) -> list[list[float]]:  # noqa: ARG001
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectoriser.fit_transform(text)
    return matrix.toarray().tolist()


@memory.cache
def _jaccard_binary_encode(text: list[str], model_name: str) -> list[list[float]]:  # noqa: ARG001
    """Binary word-presence vectors.

    Under Euclidean distance these produce ``sqrt(|A △ B|)``, i.e. the
    symmetric-difference count — a monotone proxy for Jaccard suitable
    for relative triplet ordering and Ward clustering.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectoriser = CountVectorizer(
        analyzer="word",
        lowercase=True,
        binary=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectoriser.fit_transform(text)
    return matrix.toarray().astype(float).tolist()


_ENCODERS: dict[str, Callable[[list[str], str], list[list[float]]]] = {
    "tfidf-char35": _tfidf_char_encode,
    "jaccard-binary": _jaccard_binary_encode,
}


def get_lexical_embedder(model_name: str) -> Embedder:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {model_name} not available. Available models: {AVAILABLE_MODELS}"
        )
    return partial(_ENCODERS[model_name], model_name=model_name)


def all_models() -> dict[str, Callable[[], Embedder]]:
    return {model: partial(get_lexical_embedder, model) for model in AVAILABLE_MODELS}
