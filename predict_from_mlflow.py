"""Load the Champion from the MLflow Model Registry by alias and predict on NEW data.

This script never imports transformers: prediction goes through the MLflow pyfunc model.
It does not know which version is the Champion; the alias decides.

Usage:  python predict_from_mlflow.py
        python predict_from_mlflow.py --alias challenger
"""
import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()
    uri = f"models:/{C.REGISTERED_MODEL_NAME}@{args.alias}"

    mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, args.alias)
    hf_id = mv.tags.get("hf_model_id", "unknown")
    print(f"Loading {uri}")
    print(f"  -> registered model version: {mv.version}")
    print(f"  -> Hugging Face model id   : {hf_id}")
    print(f"  -> source run              : {mv.tags.get('run_name', mv.run_id)}\n")

    model = mlflow.pyfunc.load_model(uri)
    new_df = pd.read_csv("data/new_data.csv")
    out = model.predict(new_df[["text"]])

    result = new_df.copy()
    result["prediction"] = out["prediction"].values
    result["score"] = out["score"].values
    result["alias"] = args.alias
    result["model_version"] = mv.version
    result["hf_model_id"] = hf_id

    with pd.option_context("display.max_colwidth", 60, "display.width", 140):
        print(result[["text", "expected", "prediction", "score"]].to_string(index=False))
    acc = (result["prediction"] == result["expected"]).mean()
    print(f"\nMatches my expected label on {acc*100:.0f}% of the {len(result)} new examples")

    out_path = "outputs/new_data_predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
