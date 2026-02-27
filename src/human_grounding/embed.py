from dataclasses import dataclass
from functools import partial

import polars as pl
from joblib import Memory
from tqdm import tqdm

import human_grounding.instruct_embed
import human_grounding.model2vec_embed
import human_grounding.oai as oai
import human_grounding.sentence_embed
from human_grounding.data_models import Embedder, dataclasses_to_dataframe
from human_grounding.directories import CACHE_DIR

memory = Memory(CACHE_DIR, verbose=0)


@dataclass
class EmbeddingConfig:
    dimensions: int = 256
    model: str = "text-embedding-3-large"


def batch_text(text: list[str], max_batch: int) -> list[list[str]]:
    return [text[i : i + max_batch] for i in range(0, len(text), max_batch)]


def embed_text(
    text: list[str], embedder: Embedder, max_batch: int = 1000
) -> list[list[float]]:
    outputs = []
    if len(text) < max_batch:
        return embedder(text)
    for batch in tqdm(batch_text(text, max_batch), desc="Embedding text"):
        outputs.extend(embedder(batch))
    return outputs


async def aembed_text(text: list[str], embedder: Embedder) -> list[list[float]]:
    return await embedder(text)


async def aembed_text_oai(
    text: list[str],
    client: None | oai.AsyncOpenAI = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[list[float]]:
    if client is None:
        client = oai.initialize_oai_client(is_async=True)
    if embedding_config is None:
        embedding_config = EmbeddingConfig()
    raw_embeddings = await client.embeddings.create(
        input=text, model=embedding_config.model, dimensions=embedding_config.dimensions
    )
    return [embedding.embedding for embedding in raw_embeddings.data]


@memory.cache(ignore=["client"])
def openai_embedder(
    texts: list[str],
    model: str = "text-embedding-3-large",
    num_dimensions: int = 256,
    client: oai.OpenAI | None = None,
) -> list[list[float]]:
    if client is None:
        client = oai.initialize_oai_client()
    if "002" in model:
        raw_embeddings = client.embeddings.create(input=texts, model=model)
    else:
        raw_embeddings = client.embeddings.create(
            input=texts, model=model, dimensions=num_dimensions
        )

    return [embedding.embedding for embedding in raw_embeddings.data]


def embed_dataclasses(
    dataclasses,  # noqa: ANN001
    embedder: Embedder,
    text_field: str = "text",
    batch_size: int = 1000,
) -> pl.DataFrame:
    texts = [getattr(dataclass, text_field) for dataclass in dataclasses]
    results = dataclasses_to_dataframe(dataclasses)

    embeddings = embed_text(texts, embedder, max_batch=batch_size)
    return results.with_columns(pl.Series(name="embedding", values=embeddings))


def embed_series(
    series: pl.Series,
    embedder: Embedder,
    embedding_name: str = "embedding",
) -> pl.Series:
    embeddings = embed_text(series.to_list(), embedder)
    embedding_size = len(embeddings[0])
    return pl.Series(
        name=embedding_name,
        values=embeddings,
        dtype=pl.Array(pl.Float64, shape=embedding_size),
    )


def add_embedding_column(
    df: pl.DataFrame,
    model_name: str = "text-embedding-3-large",
    text_column: str = "statement",
    embedder: Embedder = openai_embedder,
) -> pl.DataFrame:
    embeddings = embed_series(
        series=df[text_column],
        embedder=embedder,
    )
    return df.with_columns(
        embeddings,
        pl.lit(model_name).alias("embedding_model"),
    )


def get_embedder(embedding_model: str) -> Embedder:
    all_models = get_all_models()
    if embedding_model not in all_models:
        raise ValueError(
            f"Model {embedding_model} not available. Available models: {all_models.keys()}"
        )

    embedder = all_models[embedding_model]
    if embedding_model.startswith("text-embedding-"):
        return embedder
    return embedder()


def get_openai_models() -> dict[str, Embedder]:
    available_models = [
        "text-embedding-3-large",
        "text-embedding-ada-002",
        "text-embedding-3-small",
    ]
    return {model: partial(openai_embedder, model=model) for model in available_models}


def get_all_models() -> dict[str, Embedder]:
    openai_models = get_openai_models()
    sentence_transformers = human_grounding.sentence_embed.all_models()
    model2vec_models = human_grounding.model2vec_embed.all_models()
    instruct_models = human_grounding.instruct_embed.all_models()
    return {
        **openai_models,
        **sentence_transformers,
        **model2vec_models,
        **instruct_models,
    }  # ty:ignore[invalid-return-type]


if __name__ == "__main__":
    text = ["Hello, world!"]
    embeddings = embed_text(text, openai_embedder)
    embeddings
    print(embeddings)
