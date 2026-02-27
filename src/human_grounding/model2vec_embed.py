from collections.abc import Callable
from functools import partial

from joblib import Memory

from human_grounding.data_models import Embedder, SentenceEmbedder
from human_grounding.directories import CACHE_DIR

MODELS = {
    "potion-multilingual-128M": "minishlab/potion-multilingual-128M",
    "potion-base-8M": "minishlab/potion-base-8M",
    "potion-base-4M": "minishlab/potion-base-4M",
    "potion-base-2M": "minishlab/potion-base-2M",
    "M2V_base_glove": "minishlab/M2V_base_glove",
    "M2V_base_glove_subword": "minishlab/M2V_base_glove_subword",
    "M2V_base_output": "minishlab/M2V_base_output",
    "m2v-dfm-large": "rasgaard/m2v-dfm-large",
    "model2vecdk": "andersborges/model2vecdk",
    "model2vecdk-stem": "andersborges/model2vecdk-stem",
}

AVAILABLE_MODELS = set(MODELS.keys())
memory = Memory(CACHE_DIR, verbose=0)


def get_model2vec_embedder(model_name: str) -> Embedder:
    model = load_model(model_name)

    @memory.cache
    def encode(text: list[str], model_name: str) -> list[list[float]]:  # noqa: ARG001
        """Note: we add model_name to the signature to make the cache key unique"""
        return list(model.encode(text))

    return partial(encode, model_name=model_name)


def load_model(model_name: str) -> SentenceEmbedder:
    from model2vec import StaticModel

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {model_name} not available. Available models: {AVAILABLE_MODELS}"
        )
    model = StaticModel.from_pretrained(MODELS[model_name])
    return model


def all_models() -> dict[str, Callable[[], Embedder]]:
    return {model: partial(get_model2vec_embedder, model) for model in AVAILABLE_MODELS}
