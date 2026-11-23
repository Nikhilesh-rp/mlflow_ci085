"""Register Model A and Model B from the latest compare run and set the
'champion' / 'challenger' aliases according to the decision logged by evaluate.py.

No run IDs are hard-coded: the script finds the latest compare_<task> run by name.

Usage:  python register_champion.py           (register + set aliases from the decision)
        python register_champion.py --flip    (stretch goal: swap the two aliases)
"""
import argparse

import mlflow
from mlflow import MlflowClient

import config as C


def latest_compare_run(client):
    exp = client.get_experiment_by_name(C.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment '{C.EXPERIMENT_NAME}' not found. Run evaluate.py first.")
    runs = client.search_runs([exp.experiment_id],
                              filter_string=f"attributes.run_name = 'compare_{C.TASK}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit("No compare run found. Run evaluate.py first.")
    return exp, runs[0]


def child_run(client, exp, parent_id, prefix):
    runs = client.search_runs([exp.experiment_id],
                              filter_string=f"tags.mlflow.parentRunId = '{parent_id}'")
    for r in runs:
        if r.info.run_name.startswith(prefix + "_"):
            return r
    raise SystemExit(f"Could not find the {prefix}_ child run under {parent_id}")


def flip(client):
    a = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "champion")
    b = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "challenger")
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", b.version)
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", a.version)
    print(f"Flipped: champion -> v{b.version}, challenger -> v{a.version}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", action="store_true", help="swap champion/challenger aliases")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()
    if args.flip:
        flip(client)
        return

    exp, parent = latest_compare_run(client)
    winner = parent.data.tags.get("winner", "A:")[0]   # "A" or "B"
    print(f"Compare run {parent.info.run_name} ({parent.info.run_id[:8]}...) says winner = {winner}")
    print(f"Reason: {parent.data.tags.get('decision_reason')}")

    versions = {}
    for letter in ("A", "B"):
        r = child_run(client, exp, parent.info.run_id, letter)
        uri = r.data.tags["model_uri"]
        mv = mlflow.register_model(uri, C.REGISTERED_MODEL_NAME)
        hf_id = r.data.params["hf_model_id"]
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "hf_model_id", hf_id)
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "run_name", r.info.run_name)
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "macro_f1",
                                     f"{r.data.metrics['macro_f1']:.4f}")
        versions[letter] = mv.version
        print(f"Registered {r.info.run_name} ({hf_id}) as {C.REGISTERED_MODEL_NAME} v{mv.version}")

    loser = "B" if winner == "A" else "A"
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", versions[winner])
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", versions[loser])
    print(f"\nAlias champion   -> v{versions[winner]} (model {winner})")
    print(f"Alias challenger -> v{versions[loser]} (model {loser})")


if __name__ == "__main__":
    main()
