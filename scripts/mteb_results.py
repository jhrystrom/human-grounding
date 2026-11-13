"""
Contributing results to mteb
"""

import mteb
from tqdm import tqdm

from human_grounding.embed import get_openai_models
from human_grounding.instruct_embed import MODELS as INSTRUCT_MODELS
from human_grounding.model2vec_embed import MODELS as M2V_MODELS
from human_grounding.sentence_embed import MODELS as SENTENCE_MODELS


def get_mteb_embedding_names() -> list[str]:
    openai_models = [f"openai/{model_name}" for model_name in get_openai_models()]
    return [
        *SENTENCE_MODELS.values(),
        *INSTRUCT_MODELS.values(),
        *M2V_MODELS.values(),
        *openai_models,
    ]


if __name__ == "__main__":
    # 1. Initialize cache
    cache = mteb.ResultCache()
    # TODO: Update with the full list of models (pending new PR with the other dataset)
    MODELS = get_mteb_embedding_names()

    # 2. Evaluate models
    all_results = []
    for model in tqdm(MODELS):
        model_meta = mteb.get_model_meta(model)
        task = mteb.get_task("StakeholderGrounding")
        mteb.evaluate(model_meta, task, cache=cache)
        all_results.append(model_meta)

    # 3. Submit results
    results = cache.submit_results(
        all_results, create_pr=False
    )  # manual review before pushing
