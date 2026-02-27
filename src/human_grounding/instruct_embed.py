from collections.abc import Callable
from functools import partial

from joblib import Memory

from human_grounding.data_models import Embedder, SentenceEmbedder
from human_grounding.directories import CACHE_DIR, MODEL_DIR

MODELS: dict[str, str] = {
    "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "multilingual-e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "EmbeddingGemma-Scandi-300m": "emillykkejensen/EmbeddingGemma-Scandi-300m",
    "Qwen3-Embedding-Scandi-0.6B": "emillykkejensen/Qwen3-Embedding-Scandi-0.6B",
}

INSTRUCTIONS = {
    "standard": "Cluster similar statements",
    "EmbeddingGemma-Scandi-300m": "task: sentence similarity | query: ",
    "multilingual-e5-large-instruct": "Instruct: Cluster similar statements\nQuery: ",
}


AVAILABLE_MODELS = set(MODELS.keys())
memory = Memory(CACHE_DIR, verbose=0)


def get_sentence_embedder(model_name: str) -> Embedder:
    model = load_model(model_name)

    @memory.cache
    def encode(text: list[str], model_name: str) -> list[list[float]]:
        """Note: we add model_name to the signature to make the cache key unique"""
        instruction = INSTRUCTIONS.get(model_name, INSTRUCTIONS["standard"])
        return list(model.encode(text, prompt=instruction))

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
    model_name = "EmbeddingGemma-Scandi-300m"
    test_embedder = get_sentence_embedder(model_name)
    test_sentences = ["This is a test sentence.", "This is another sentence."]
    embeddings = test_embedder(test_sentences)
