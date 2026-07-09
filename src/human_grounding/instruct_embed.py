from collections.abc import Callable
from functools import partial
from typing import NamedTuple

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
# Instruction variations for robustness experiments
# ---------------------------------------------------------------------------
# We probe two independent axes of the instruction:
#   1. *Prompt variation* — how the task itself is phrased (PROMPT_VARIATIONS).
#   2. *Dataset context*   — an optional domain prefix that tells the model which
#                            dataset the statements come from (DATASET_INSTRUCTION_PREFIX).
# Crossing them lets a single plot compare prompt-robustness AND the effect of
# adding dataset context. Each (base, variant, context) triple is turned into a
# *distinct, deterministic* model name so joblib's caches — which are keyed on
# the model name — never collide. ``PROMPT_VARIATIONS[DEFAULT_VARIANT]`` with no
# context reproduces the historical "standard" instruction.

VARIANT_SEPARATOR = "__prompt-"
CONTEXT_SEPARATOR = "__ctx-"
DEFAULT_VARIANT = "cluster"

# Core natural-language instruction per variant key.
PROMPT_VARIATIONS: dict[str, str] = {
    "cluster": "Cluster similar statements",
    "similarity": "Retrieve semantically similar statements",
    "grouping": "Group statements that express the same idea",
    "topic": "Identify statements about the same topic",
    "meaning": "Find statements that have the same meaning",
}

# Optional domain prefix prepended to the instruction, keyed by dataset name.
# A model carrying one of these is meant to be evaluated *only* on that dataset.
DATASET_INSTRUCTION_PREFIX: dict[str, str] = {
    "rai": "The statements are about responsible AI. ",
    "welfare": "The statements are about social welfare policy. ",
}

# Per-base-model prompt template. ``{instruction}`` is filled with the variant's
# core instruction above (optionally with a dataset-context prefix). Models
# absent here use the bare instruction, which is how sentence-transformers
# expects the ``prompt=`` argument for most models.
PROMPT_TEMPLATES: dict[str, str] = {
    "multilingual-e5-large-instruct": "Instruct: {instruction}\nQuery: ",
    "EmbeddingGemma-Scandi-300m": "task: {instruction} | query: ",
}


class ModelSpec(NamedTuple):
    """Decoded parts of a (possibly variant) model name.

    ``variant`` and ``context`` are ``None`` when absent — a plain base-model
    name decodes to ``ModelSpec(base, None, None)`` (legacy behaviour).
    """

    base: str
    variant: str | None
    context: str | None


def make_variant_name(base_model: str, variant: str, context: str | None = None) -> str:
    """Deterministic model name encoding (base_model, prompt variant, context)."""
    name = f"{base_model}{VARIANT_SEPARATOR}{variant}"
    if context is not None:
        name += f"{CONTEXT_SEPARATOR}{context}"
    return name


def parse_variant_name(model_name: str) -> ModelSpec:
    """Decode a model name into its ``ModelSpec`` parts."""
    context: str | None = None
    if CONTEXT_SEPARATOR in model_name:
        model_name, context = model_name.split(CONTEXT_SEPARATOR, 1)
    if VARIANT_SEPARATOR in model_name:
        base, variant = model_name.split(VARIANT_SEPARATOR, 1)
        return ModelSpec(base, variant, context)
    return ModelSpec(model_name, None, context)


def variant_model_names(base_models: set[str] | None = None) -> list[str]:
    """Every variant model name: each base and prompt variation, generic and
    with each dataset context."""
    bases = AVAILABLE_MODELS if base_models is None else base_models
    names: list[str] = []
    for base in sorted(bases):
        for variant in PROMPT_VARIATIONS:
            names.append(make_variant_name(base, variant))
            for context in DATASET_INSTRUCTION_PREFIX:
                names.append(make_variant_name(base, variant, context))
    return names


def resolve_instruction(model_name: str) -> str:
    """Return the full prompt string for a base or variant model name."""
    base, variant, context = parse_variant_name(model_name)
    if variant is None:
        # Legacy base model: preserve the exact historical instruction (and thus
        # its existing cache entries).
        return INSTRUCTIONS.get(base, INSTRUCTIONS["standard"])
    if variant not in PROMPT_VARIATIONS:
        raise ValueError(
            f"Unknown prompt variant {variant!r}. "
            f"Available: {sorted(PROMPT_VARIATIONS)}"
        )
    core = PROMPT_VARIATIONS[variant]
    if context is not None:
        if context not in DATASET_INSTRUCTION_PREFIX:
            raise ValueError(
                f"Unknown dataset context {context!r}. "
                f"Available: {sorted(DATASET_INSTRUCTION_PREFIX)}"
            )
        core = f"{DATASET_INSTRUCTION_PREFIX[context]}{core}"
    template = PROMPT_TEMPLATES.get(base, "{instruction}")
    return template.format(instruction=core)


def get_sentence_embedder(model_name: str) -> Embedder:
    base_model = parse_variant_name(model_name).base
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

    base_model = parse_variant_name(model_name).base
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
