from collections.abc import Callable
from functools import partial

from joblib import Memory

from human_grounding.data_models import Embedder, SentenceEmbedder
from human_grounding.directories import CACHE_DIR, MODEL_DIR

MODELS: dict[str, str] = {
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "snowflake-arctic-embed-l-v2.0": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "nb-sbert-base": "NbAiLab/nb-sbert-base",
    "dfm-sentence-encoder-large": "KennethEnevoldsen/dfm-sentence-encoder-large",
    "mmBERTscandi-base-embedding": "emillykkejensen/Qwen3-Embedding-Scandi-0.6B",
    # "dfm-sentence-encoder-medium": "KennethEnevoldsen/dfm-sentence-encoder-medium",  # Currently fails to load
    "LaBSE": "sentence-transformers/LaBSE",
    "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "static-similarity-mrl-multilingual-v1": "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
}

AVAILABLE_MODELS = set(MODELS.keys())
memory = Memory(CACHE_DIR, verbose=0)


def get_sentence_embedder(model_name: str) -> Embedder:
    model = load_model(model_name)

    @memory.cache
    def encode(text: list[str], model_name: str) -> list[list[float]]:  # noqa: ARG001
        """Note: we add model_name to the signature to make the cache key unique"""
        return list(model.encode(text))

    return partial(encode, model_name=model_name)


def load_model(model_name: str) -> SentenceEmbedder:
    from sentence_transformers import SentenceTransformer

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {model_name} not available. Available models: {AVAILABLE_MODELS}"
        )
    trust_remote_code = model_name == "gte-multilingual-base"
    model_path = MODELS[model_name]
    model = SentenceTransformer(model_path, trust_remote_code=trust_remote_code)
    return model


def all_models() -> dict[str, Callable[[], Embedder]]:
    return {model: partial(get_sentence_embedder, model) for model in AVAILABLE_MODELS}


if __name__ == "__main__":
    model_name = "dfm-sentence-encoder-medium"
    test_embedder = get_sentence_embedder(model_name)
    test_sentences = ["This is a test sentence.", "This is another sentence."]
    embeddings = test_embedder(test_sentences)
