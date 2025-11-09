# scripts/promote_best.py
import argparse, os, json
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def main(args):
    # Resolve tracking URI: arg > env > default
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name  = args.model_name
    train_date  = args.train_date.strip()           # "YYYY-MM-DD"
    metric_key  = args.metric
    stage       = args.stage
    archive_old = bool(args.archive_existing)
    dry_run     = bool(args.dry_run)

    # 1) Find experiment
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"[PROMOTE] Experiment '{args.experiment}' not found at {tracking_uri}.")
        return 1

    # 2) Search runs for this train_date, order by the chosen metric desc
    flt = (
        f"tags.train_date = '{train_date}' "
        "and attributes.status = 'FINISHED'"
    )

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=flt,
        order_by=[f"metrics.{metric_key} DESC"],  # ordering is fine even if some are NaN
        max_results=200,
    )

    metric_series = pd.to_numeric(runs.get(f"metrics.{metric_key}"), errors="coerce")
    runs = runs[metric_series.notna()]

    if runs.empty:
        print(f"[PROMOTE] No runs found for train_date={train_date} with metric '{metric_key}'.")
        return 2

    best = runs.iloc[0]
    run_id = best.run_id
    best_metric = float(best[f"metrics.{metric_key}"])
    print(f"[PROMOTE] Best run for {train_date}: run_id={run_id} | {metric_key}={best_metric:.6f}")

    # Optional guardrails (e.g., don't promote if test AUC is too low)
    if args.min_test_auc is not None:
        mt = best.get("metrics.auc_test")
        if mt is None or float(mt) < args.min_test_auc:
            print(f"[PROMOTE] Guard failed: auc_test={mt} < min_test_auc={args.min_test_auc}. Aborting.")
            return 3

    # Collect a compact metrics/params snapshot for description
    keys_want = [
        "auc_train","auc_test","auc_oot",
        "gini_train","gini_test","gini_oot",
        "accuracy_test","precision_weighted_test","recall_weighted_test","f1_weighted_test",
    ]
    metrics_snapshot = {}
    for k in keys_want:
        v = best.get(f"metrics.{k}")
        if v is not None:
            metrics_snapshot[k] = float(v)

    params_snapshot = {}
    for col in best.index:
        if col.startswith("params."):
            params_snapshot[col.split("params.", 1)[1]] = best[col]

    # 3) Ensure there is a Model Version for this run
    # Discover the logged artifact path from the run's tags (robust to any name)
    run = client.get_run(run_id)
    hist_tag = run.data.tags.get("mlflow.log-model.history")

    artifact_path = None
    if hist_tag:
        try:
            hist = json.loads(hist_tag)
            # Prefer the most recent logged model that has actual artifacts
            for entry in reversed(hist):
                cand = entry.get("artifact_path")
                if not cand:
                    continue
                # verify this artifact path exists for this run
                try:
                    client.list_artifacts(run_id, cand)
                    artifact_path = cand
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"[PROMOTE] Could not parse mlflow.log-model.history: {e}")

    # Fallback to the conventional 'model' path if nothing found
    if not artifact_path:
        artifact_path = "model"

    src_uri = f"runs:/{run_id}/{artifact_path}"
    print(f"[PROMOTE] Using artifact_path='{artifact_path}' -> {src_uri}")

    # If already registered for this run, reuse it (idempotent)
    existing = [
        mv for mv in client.search_model_versions(f"name='{model_name}'")
        if mv.run_id == run_id and mv.source == src_uri
    ]
    if existing:
        mv = sorted(existing, key=lambda m: int(m.version))[-1]
        print(f"[PROMOTE] Reusing existing model version: {model_name} v{mv.version}")
    else:
        if dry_run:
            print(f"[PROMOTE][DRY] Would register model from source: {src_uri}")
            return 0
        mv = mlflow.register_model(src_uri, model_name)
        print(f"[PROMOTE] Registered: {model_name} v{mv.version}")

    # 4) Update model version description with metrics/params
    desc = {
        "train_date": train_date,
        "opt_metric": metric_key,
        "opt_metric_value": best_metric,
        "run_id": run_id,
        "metrics": metrics_snapshot,
        "params": params_snapshot,
    }
    client.update_model_version(
        name=model_name,
        version=mv.version,
        description=json.dumps(desc, indent=2, sort_keys=True),
    )

    # 5) Transition stage
    if dry_run:
        print(f"[PROMOTE][DRY] Would transition {model_name} v{mv.version} -> {stage} (archive_existing={archive_old})")
        return 0

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
        archive_existing_versions=archive_old,
    )
    print(f"[PROMOTE] {model_name} v{mv.version} -> {stage} (archive_existing={archive_old})")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Promote best MLflow run (by metric) for a given train date")
    ap.add_argument("--tracking-uri", default=None, help="Fallback to $MLFLOW_TRACKING_URI or http://mlflow:5000")
    ap.add_argument("--experiment", default="credit_risk_training")
    ap.add_argument("--model-name", default="credit_risk_model")
    ap.add_argument("--train-date", required=True, help="YYYY-MM-DD (matches tags.train_date)")
    ap.add_argument("--metric", default="auc_oot", help="Metric key to maximize (e.g., auc_oot, f1_weighted_test)")
    ap.add_argument("--min-test-auc", type=float, default=None, help="Guard: require auc_test >= this")
    ap.add_argument("--stage", default="Production", choices=["Staging","Production","Archived","None"])
    ap.add_argument("--archive-existing", type=int, default=1, help="1/0 to archive existing versions in target stage")
    ap.add_argument("--dry-run", type=int, default=0, help="1=plan only; 0=apply")
    args = ap.parse_args()
    raise SystemExit(main(args))
