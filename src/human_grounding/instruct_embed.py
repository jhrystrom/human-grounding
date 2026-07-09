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


# ---------------------------------------------------------------------------
# Prompt (instruction) variations for robustness experiments
# ---------------------------------------------------------------------------
# To test how sensitive alignment is to the phrasing of the task instruction,
# we run each base model under several instruction phrasings. Each variation is
# turned into a *distinct, deterministic* model name (``{base}{SEP}{variant}``)
# so joblib's caches — which are keyed on the model name — never collide across
# variations. ``PROMPT_VARIATIONS[DEFAULT_VARIANT]`` reproduces the historical
# "standard" instruction, so the default variant matches the legacy base model.

VARIANT_SEPARATOR = "__prompt-"
DEFAULT_VARIANT = "cluster"

# Core natural-language instruction per variant key.
PROMPT_VARIATIONS: dict[str, str] = {
    "cluster": "Cluster similar statements",
    "similarity": "Retrieve semantically similar statements",
    "grouping": "Group statements that express the same idea",
    "topic": "Identify statements about the same topic",
    "meaning": "Represent the statement for meaning-based comparison",
}

# Per-base-model prompt template. ``{instruction}`` is filled with the variant's
# core instruction above. Models absent here use the bare instruction, which is
# how sentence-transformers expects the ``prompt=`` argument for most models.
PROMPT_TEMPLATES: dict[str, str] = {
    "multilingual-e5-large-instruct": "Instruct: {instruction}\nQuery: ",
    "EmbeddingGemma-Scandi-300m": "task: {instruction} | query: ",
}


def make_variant_name(base_model: str, variant: str) -> str:
    """Deterministic model name encoding a (base_model, prompt variant) pair."""
    return f"{base_model}{VARIANT_SEPARATOR}{variant}"


def parse_variant_name(model_name: str) -> tuple[str, str | None]:
    """Split a model name into ``(base_model, variant)``.

    ``variant`` is ``None`` for a plain base-model name (legacy behaviour).
    """
    if VARIANT_SEPARATOR in model_name:
        base, variant = model_name.split(VARIANT_SEPARATOR, 1)
        return base, variant
    return model_name, None


def variant_model_names(base_models: set[str] | None = None) -> list[str]:
    """All ``{base}{SEP}{variant}`` names across base models and variations."""
    bases = AVAILABLE_MODELS if base_models is None else base_models
    return [
        make_variant_name(base, variant)
        for base in sorted(bases)
        for variant in PROMPT_VARIATIONS
    ]


def resolve_instruction(model_name: str) -> str:
    """Return the full prompt string for a base or variant model name."""
    base, variant = parse_variant_name(model_name)
    if variant is None:
        # Legacy base model: preserve the exact historical instruction (and thus
        # its existing cache entries).
        return INSTRUCTIONS.get(base, INSTRUCTIONS["standard"])
    if variant not in PROMPT_VARIATIONS:
        raise ValueError(
            f"Unknown prompt variant {variant!r}. "
            f"Available: {sorted(PROMPT_VARIATIONS)}"
        )
    template = PROMPT_TEMPLATES.get(base, "{instruction}")
    return template.format(instruction=PROMPT_VARIATIONS[variant])


def get_sentence_embedder(model_name: str) -> Embedder:
    base_model, _ = parse_variant_name(model_name)
    model = load_model(base_model)
    instruction = resolve_instruction(model_name)

    @memory.cache
    def encode(text: list[str], model_name: str) -> list[list[float]]:  # noqa: ARG001
        """Note: we add model_name to the signature to make the cache key unique.

        ``model_name`` here is the full (possibly variant) name, so distinct
        prompt variations get distinct cache entries deterministically.
        """
        return list(model.encode(text, prompt=instruction))

    return partial(encode, model_name=model_name)


def load_model(model_name: str) -> SentenceEmbedder:
    from sentence_transformers import SentenceTransformer

    base_model, _ = parse_variant_name(model_name)
    if base_model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {base_model} not available. Available models: {AVAILABLE_MODELS}"
        )
    trust_remote_code = base_model == "gte-multilingual-base"
    model_path = MODELS[base_model]
    model = SentenceTransformer(model_path, trust_remote_code=trust_remote_code)
    return model


def all_models() -> dict[str, Callable[[], Embedder]]:
    """Base models plus every prompt-variation variant model."""
    names = [*sorted(AVAILABLE_MODELS), *variant_model_names()]
    return {name: partial(get_sentence_embedder, name) for name in names}


if __name__ == "__main__":
    model_name = "EmbeddingGemma-Scandi-300m"
    test_embedder = get_sentence_embedder(model_name)
    test_sentences = ["This is a test sentence.", "This is another sentence."]
    embeddings = test_embedder(test_sentences)
