"""Run Model A (Champion) and Model B (Challenger) on the same SST-2 inputs and log
everything to MLflow: one run per model, nested inside a compare_sentiment summary run.

Usage:  python evaluate.py            (full 200 examples)
        python evaluate.py --n 10     (quick debug run)
"""
import argparse
import json
import os
import platform
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import config as C
from hf_wrapper import HFSentimentModel

LABELS = ["negative", "positive"]
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def library_versions():
    vers = {"mlflow": mlflow.__version__, "python": platform.python_version()}
    for lib in ("transformers", "torch"):
        try:
            vers[lib] = __import__(lib).__version__
        except ImportError:
            vers[lib] = "not-installed"
    return vers


def load_eval_set(n):
    """Sample n labelled sentences from SST-2 validation with a fixed seed and save them."""
    from datasets import load_dataset

    ds = load_dataset(C.DATASET_NAME, split=C.DATASET_SPLIT)
    df = ds.to_pandas()[["idx", "sentence", "label"]]
    df = df.sample(n=n, random_state=C.SEED).reset_index(drop=True)
    df = df.rename(columns={"idx": "source_idx", "sentence": "text"})
    df["reference"] = df["label"].map({0: "negative", 1: "positive"})
    df = df[["source_idx", "text", "reference"]]
    df.to_csv("data/eval_inputs.csv", index=False)
    return df


def log_model_compat(python_model, input_example, signature, extra_pip):
    """MLflow 3 uses name=, MLflow 2 uses artifact_path=. Try the new one first."""
    kwargs = dict(python_model=python_model, code_paths=["hf_wrapper.py"],
                  input_example=input_example, signature=signature,
                  pip_requirements=extra_pip)
    try:
        return mlflow.pyfunc.log_model(name="model", **kwargs)
    except TypeError:
        return mlflow.pyfunc.log_model(artifact_path="model", **kwargs)


def compute_metrics(ref, pred):
    return {
        "accuracy": accuracy_score(ref, pred),
        "macro_f1": f1_score(ref, pred, average="macro", labels=LABELS, zero_division=0),
        "macro_precision": precision_score(ref, pred, average="macro", labels=LABELS, zero_division=0),
        "macro_recall": recall_score(ref, pred, average="macro", labels=LABELS, zero_division=0),
    }


def plot_confusion(ref, pred, title, path):
    cm = confusion_matrix(ref, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def run_one(role_letter, spec, role, eval_df, versions, device):
    """Predict with one model, then log params/metrics/tags/artifacts/model in a child run."""
    short = spec["short_name"]
    run_name = f"{role_letter}_{C.TASK}_{short}"
    print(f"\n=== {run_name}  ({spec['hf_model_id']}) ===")

    model = HFSentimentModel(spec["hf_model_id"], C.BATCH_SIZE, C.MAX_LENGTH)
    model.load_context(None)          # downloads/loads the HF pipeline once
    inputs = eval_df[["text"]]
    model.predict(None, inputs.head(2))  # warm-up so the first batch is not counted as slow

    start = time.perf_counter()
    out = model.predict(None, inputs)
    total_s = time.perf_counter() - start

    preds = eval_df[["source_idx", "text", "reference"]].copy()
    preds["prediction"] = out["prediction"].values
    preds["score"] = out["score"].values
    preds["correct"] = (preds["prediction"] == preds["reference"]).astype(int)
    pred_path = f"outputs/predictions_{short}.csv"
    preds.to_csv(pred_path, index=False)

    metrics = compute_metrics(preds["reference"], preds["prediction"])
    metrics["neutral_predictions"] = int((preds["prediction"] == "neutral").sum())
    metrics["avg_latency_ms"] = 1000 * total_s / len(preds)
    metrics["total_inference_s"] = total_s

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params({
            "hf_model_id": spec["hf_model_id"],
            "task": C.TASK,
            "role": role,
            "batch_size": C.BATCH_SIZE,
            "max_length": C.MAX_LENGTH,
            "eval_samples": len(preds),
            "seed": C.SEED,
            "device": device,
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "student_usn": C.STUDENT_USN,
            "library_versions": json.dumps(versions),
            "dataset_name": f"{C.DATASET_NAME}:{C.DATASET_SPLIT}",
            "hf_model_id": spec["hf_model_id"],
        })

        metrics_file = f"outputs/metrics_summary_{short}.json"
        with open(metrics_file, "w") as f:
            json.dump({"run_name": run_name, "hf_model_id": spec["hf_model_id"], **metrics}, f, indent=2)

        cm_path = f"outputs/confusion_matrix_{short}.png"
        fig = plot_confusion(preds["reference"], preds["prediction"], run_name, cm_path)
        plt.close(fig)

        mlflow.log_artifact(pred_path, "predictions")
        mlflow.log_artifact("data/eval_inputs.csv", "data")
        mlflow.log_artifact(metrics_file)
        mlflow.log_artifact(cm_path, "plots")

        # Log the wrapped model (fresh instance so the loaded pipeline is not pickled)
        sample = inputs.head(3)
        signature = infer_signature(sample, out.head(3))
        reqs = [f"mlflow=={mlflow.__version__}", "transformers", "torch", "pandas"]
        info = log_model_compat(HFSentimentModel(spec["hf_model_id"], C.BATCH_SIZE, C.MAX_LENGTH),
                                sample, signature, reqs)
        mlflow.set_tag("model_uri", info.model_uri)
        print(f"  accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
              f"avg_latency_ms={metrics['avg_latency_ms']:.1f}")
        print(f"  model URI: {info.model_uri}")
        return {"run_id": run.info.run_id, "run_name": run_name, "model_uri": info.model_uri,
                "metrics": metrics, "preds": preds, "short": short}


def decide(a, b):
    """Apply the Champion rule from config.py / README. Returns (winner, reason, delta)."""
    pa, pb = a["metrics"][C.PRIMARY_METRIC], b["metrics"][C.PRIMARY_METRIC]
    delta = pb - pa
    ratio = b["metrics"]["avg_latency_ms"] / max(a["metrics"]["avg_latency_ms"], 1e-9)
    if ratio > C.MAX_LATENCY_RATIO:
        return "A", f"Challenger rejected by guardrail: latency {ratio:.1f}x > {C.MAX_LATENCY_RATIO}x", delta, ratio
    if delta >= C.MARGIN:
        return "B", f"Challenger beats Champion by {delta*100:.2f} pp >= {C.MARGIN*100:.1f} pp margin", delta, ratio
    return "A", f"Difference {delta*100:.2f} pp is below the {C.MARGIN*100:.1f} pp margin, Champion stays", delta, ratio


def comparison_chart(a, b, path):
    names = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [a["metrics"][n] for n in names], 0.4, label=f"A (Champion) {a['short']}")
    ax.bar(x + 0.2, [b["metrics"][n] for n in names], 0.4, label=f"B (Challenger) {b['short']}")
    ax.set_xticks(x, names)
    lo = min(min(a["metrics"][n], b["metrics"][n]) for n in names)
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Champion vs Challenger on the same eval set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def latency_chart(a, b, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals = [a["metrics"]["avg_latency_ms"], b["metrics"]["avg_latency_ms"]]
    ax.bar([f"A {a['short']}", f"B {b['short']}"], vals, color=["#4C78A8", "#F58518"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f} ms", ha="center", va="bottom")
    ax.set_ylabel("avg latency per example (ms, lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.EVAL_SAMPLES, help="number of eval examples")
    args = ap.parse_args()

    set_seed(C.SEED)
    mlflow.set_tracking_uri(C.TRACKING_URI)
    mlflow.set_experiment(C.EXPERIMENT_NAME)
    versions = library_versions()
    device = "cpu"

    eval_df = load_eval_set(args.n)
    print(f"Saved {len(eval_df)} evaluation examples to data/eval_inputs.csv")

    with mlflow.start_run(run_name=f"compare_{C.TASK}") as parent:
        a = run_one("A", C.MODEL_A, "champion", eval_df, versions, device)
        b = run_one("B", C.MODEL_B, "challenger", eval_df, versions, device)

        winner, reason, delta, ratio = decide(a, b)
        win = a if winner == "A" else b
        print(f"\nDecision: {reason}\nWinner: {winner} ({win['run_name']})")

        mlflow.log_params({
            "primary_metric": C.PRIMARY_METRIC,
            "margin": C.MARGIN,
            "max_latency_ratio": C.MAX_LATENCY_RATIO,
            "eval_samples": len(eval_df),
            "model_a": C.MODEL_A["hf_model_id"],
            "model_b": C.MODEL_B["hf_model_id"],
        })
        mlflow.log_metrics({
            "delta_primary_metric": delta,
            "delta_accuracy": b["metrics"]["accuracy"] - a["metrics"]["accuracy"],
            "latency_ratio_b_over_a": ratio,
            f"a_{C.PRIMARY_METRIC}": a["metrics"][C.PRIMARY_METRIC],
            f"b_{C.PRIMARY_METRIC}": b["metrics"][C.PRIMARY_METRIC],
        })
        mlflow.set_tags({
            "winner": f"{winner}:{win['run_name']}",
            "winner_hf_model_id": (C.MODEL_A if winner == "A" else C.MODEL_B)["hf_model_id"],
            "decision_reason": reason,
            "student_usn": C.STUDENT_USN,
            "dataset_name": f"{C.DATASET_NAME}:{C.DATASET_SPLIT}",
        })

        # Comparison plots
        f1 = comparison_chart(a, b, "outputs/compare_metrics.png")
        mlflow.log_figure(f1, "plots/compare_metrics.png"); plt.close(f1)
        f2 = latency_chart(a, b, "outputs/compare_latency.png")
        mlflow.log_figure(f2, "plots/compare_latency.png"); plt.close(f2)

        # Disagreements between the two models (for qualitative analysis)
        both = a["preds"][["source_idx", "text", "reference"]].copy()
        both["pred_A"], both["score_A"] = a["preds"]["prediction"], a["preds"]["score"]
        both["pred_B"], both["score_B"] = b["preds"]["prediction"], b["preds"]["score"]
        dis = both[both["pred_A"] != both["pred_B"]].copy()
        dis["who_was_right"] = np.where(dis["pred_A"] == dis["reference"], "A",
                                np.where(dis["pred_B"] == dis["reference"], "B", "neither"))
        dis["comment"] = ""
        dis.to_csv("outputs/disagreements.csv", index=False)
        print(f"Models disagree on {len(dis)} of {len(both)} examples -> outputs/disagreements.csv")

        summary = {
            "experiment": C.EXPERIMENT_NAME,
            "eval_samples": len(eval_df),
            "rule": {"primary_metric": C.PRIMARY_METRIC, "direction": "higher is better",
                     "margin": C.MARGIN, "max_latency_ratio": C.MAX_LATENCY_RATIO},
            "A": {"run_name": a["run_name"], "hf_model_id": C.MODEL_A["hf_model_id"], **a["metrics"]},
            "B": {"run_name": b["run_name"], "hf_model_id": C.MODEL_B["hf_model_id"], **b["metrics"]},
            "delta_primary_metric": delta,
            "latency_ratio_b_over_a": ratio,
            "disagreements": int(len(dis)),
            "winner": winner,
            "decision_reason": reason,
        }
        with open("outputs/metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        for p in ["outputs/metrics_summary.json", "outputs/disagreements.csv",
                  "data/eval_inputs.csv", "README.md", "requirements.txt"]:
            if os.path.exists(p):
                mlflow.log_artifact(p)

    print("\nDone. Open http://127.0.0.1:5000 to see the runs, then run register_champion.py")


if __name__ == "__main__":
    main()
