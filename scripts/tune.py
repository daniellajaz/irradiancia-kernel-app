"""Etapa 2 fuera del notebook (misma lógica y misma caché results/tuning_results.json)."""
import json, os, sys, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT); warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402
from fkernel_lab.experiment import DATASETS, Spec, add_selection_score, holdout_split, load_dataset, tune_spec  # noqa: E402

grid = add_selection_score(pd.read_csv("results/grid_results.csv"))
g = grid.sort_values("cv_score", ascending=False).drop_duplicates(subset=["dataset", "model", "cv_score", "test_score"])
cand = g.groupby(["dataset", "model"]).head(2)
path = "results/tuning_results.json"
tuned = json.load(open(path)) if os.path.exists(path) else {}
frames = {n: load_dataset(n) for n in DATASETS}
for _, row in cand.iterrows():
    key = f"{row.dataset}|{row.spec_id}"
    if key in tuned:
        continue
    d = frames[row.dataset]; tr, te = holdout_split(d)
    best, score = tune_spec(Spec.from_id(row.spec_id), d, tr, n_iter=20)
    tuned[key] = {"params": {k: (list(v) if isinstance(v, tuple) else (v.item() if hasattr(v, "item") else v))
                             for k, v in best.items()}, "search_score": score}
    json.dump(tuned, open(path, "w"), indent=1)
    print(key, tuned[key], flush=True)
