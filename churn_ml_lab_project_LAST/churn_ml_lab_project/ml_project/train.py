"""Command-line training: python train.py  (imports ml_pipeline so pickles load inside Streamlit)."""
import json
import pandas as pd
import ml_pipeline as mp

meta, results, _ = mp.run_experiment("data/churn.csv", "Churn")
print(json.dumps(meta["report"], indent=2, default=str))
print(pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1"]].astype(float).round(3))
print("Best model:", meta["best_model"])
