"""
Contributing results to mteb
"""

import mteb
from tqdm import tqdm

# 1. Initialize cache
cache = mteb.ResultCache()
# TODO: Update with the full list of models (pending new PR with the other dataset)
MODELS = ["minishlab/potion-base-4M", "intfloat/multilingual-e5-small"]

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
