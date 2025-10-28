# scripts/promote_best.py
import argparse
import os, mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

def main(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    model_name = args.model_name
    train_date = args.train_date  # "YYYY-MM-DD"

    # Find the best same-day candidate by OOT AUC among all registered runs
    # We search runs in the experiment; if you prefer, search model versions instead.
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"[PROMOTE] Experiment '{args.experiment}' not found.")
        return

    flt = f'tags.train_date = "{train_date}"'
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=flt,
        order_by=["metrics.auc_oot DESC"],
        max_results=1,
    )
    if runs.empty:
        print(f"[PROMOTE] No runs found for train_date={train_date}")
        return

    best = runs.iloc[0]
    run_id = best.run_id
    auc = float(best["metrics.auc_oot"])
    print(f"[PROMOTE] Best same-day run: {run_id} (auc_oot={auc:.4f})")

    # Create a new Model Version from this run (points to artifact 'model')
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    print(f"[PROMOTE] Registered as {model_name} v{mv.version}")

    # Move it to Production and archive previous versions
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"[PROMOTE] {model_name} v{mv.version} -> Production (archived old Production)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default="http://mlflow:5000")
    ap.add_argument("--experiment", default="credit_risk_training")
    ap.add_argument("--model-name", default="credit_risk_model")
    ap.add_argument("--train-date", required=True)  # "{{ ds }}"
    args = ap.parse_args()
    main(args)
