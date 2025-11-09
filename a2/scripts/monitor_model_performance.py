# scripts/monitor_model_performance.py
import os, argparse, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime
from scipy.stats import ks_2samp

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _month_floor(d: date) -> date:
    return date(d.year, d.month, 1)

def _coerce_date(x):
    if isinstance(x, date):
        return x
    if x is None:
        return datetime.utcnow().date()
    return pd.to_datetime(x, errors="coerce").date()

def _month_floor_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp().dt.date

def _summarize_predictions(df: pd.DataFrame) -> dict:
    s = pd.to_numeric(df["default_proba"], errors="coerce").dropna()
    return {
        "sample_size": int(len(s)),
        "default_proba_min": float(s.min()) if len(s) else None,
        "default_proba_max": float(s.max()) if len(s) else None,
        "default_proba_mean": float(s.mean()) if len(s) else None,
        "default_proba_std": float(s.std()) if len(s) else None,
    }

def _compute_psi(base_scores, curr_scores, buckets=10):
    """Compute Population Stability Index (PSI)."""
    try:
        base_scores = np.array(base_scores, dtype=float)
        curr_scores = np.array(curr_scores, dtype=float)
        quantiles = np.percentile(base_scores, np.linspace(0, 100, buckets + 1))
        quantiles[0], quantiles[-1] = -np.inf, np.inf
        base_hist, _ = np.histogram(base_scores, bins=quantiles)
        curr_hist, _ = np.histogram(curr_scores, bins=quantiles)
        base_perc = base_hist / np.sum(base_hist)
        curr_perc = curr_hist / np.sum(curr_hist)
        psi = np.sum((curr_perc - base_perc) * np.log((curr_perc + 1e-8) / (base_perc + 1e-8)))
        return float(psi)
    except Exception as e:
        print(f"[WARN] PSI computation failed: {e}")
        return None

def _compute_ks(base_scores, curr_scores):
    """Compute KS statistic between two distributions."""
    try:
        ks_stat, _ = ks_2samp(base_scores, curr_scores)
        return float(ks_stat)
    except Exception as e:
        print(f"[WARN] KS computation failed: {e}")
        return None

def _compute_binary_metrics(y_true, scores, thr: float = 0.5) -> dict:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score,
        recall_score, f1_score, confusion_matrix
    )
    y_true = pd.Series(y_true).astype(int).values
    s = pd.to_numeric(pd.Series(scores), errors="coerce").fillna(0).astype(float).values
    y_pred = (s >= thr).astype(int)

    out = {}
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, s))
    except Exception:
        out["auc_roc"] = None
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    return out

# --------------------------------------------------------------------
# MLflow Integration
# --------------------------------------------------------------------
def _get_or_create_experiment(experiment_name: str) -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"[MLFLOW] Creating new experiment: {experiment_name}")
        return client.create_experiment(name=experiment_name)
    return exp.experiment_id

def _log_monitoring_run(experiment_name, run_id, metrics, summary, tags):
    exp_id = _get_or_create_experiment(experiment_name)
    run_name = f"monitor_{tags.get('monitor_month')}"
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_metrics({**metrics, **summary})
        print(f"[MLFLOW] Monitoring logged for {tags['monitor_month']} to run {run.info.run_id}")

# --------------------------------------------------------------------
# Main Logic
# --------------------------------------------------------------------
def main(args):
    from monitor_model_performance import _load_last_month_predictions_from_mlflow, _load_labels_for_month

    end_date = _coerce_date(args.end_date)
    last_month = _month_floor(end_date)
    prev_month = _month_floor((pd.Timestamp(last_month) - pd.offsets.MonthBegin(1)).date())

    # --- Load current predictions and labels ---
    pred_df = _load_last_month_predictions_from_mlflow(
        experiment_name=args.mlflow_experiment,
        artifact_name=args.mlflow_artifact_name,
        end_date=end_date,
    )
    if pred_df is None or pred_df.empty:
        print("[MONITOR] No predictions found; skipping.")
        return 0

    lab_df = _load_labels_for_month(args.gold_labels_dir, last_month)
    if lab_df is None or lab_df.empty:
        print("[MONITOR] No labels found; skipping.")
        return 0

    joined = pred_df.merge(
        lab_df[["Customer_ID", "snapshot_date", "label"]],
        on=["Customer_ID", "snapshot_date"],
        how="inner",
    )
    if joined.empty:
        print("[MONITOR] No joined rows; skipping.")
        return 0

    # --- compute metrics ---
    metrics = _compute_binary_metrics(joined["label"], joined["default_proba"], thr=args.threshold)
    summary = _summarize_predictions(pred_df)

    # --- compute drift metrics (vs previous month) ---
    prev_pred_df = _load_last_month_predictions_from_mlflow(
        experiment_name=args.mlflow_experiment,
        artifact_name=args.mlflow_artifact_name,
        end_date=prev_month,
    )
    if prev_pred_df is not None and len(prev_pred_df):
        s_prev = pd.to_numeric(prev_pred_df["default_proba"], errors="coerce").dropna()
        s_curr = pd.to_numeric(pred_df["default_proba"], errors="coerce").dropna()
        psi_val = _compute_psi(s_prev, s_curr)
        ks_val = _compute_ks(s_prev, s_curr)
        metrics["psi_stat"] = psi_val
        metrics["ks_stat"] = ks_val
        print(f"[DRIFT] PSI={psi_val:.4f} | KS={ks_val:.4f}")
    else:
        print("[DRIFT] Previous month predictions not found; PSI/KS skipped.")

    # --- log all to MLflow ---
    _log_monitoring_run(
        experiment_name=args.mlflow_experiment,
        run_id="unknown",
        metrics=metrics,
        summary=summary,
        tags={
            "monitor_month": last_month.isoformat(),
            "source": "monitor_script",
            "mlflow_experiment": args.mlflow_experiment,
            "mlflow_artifact_name": args.mlflow_artifact_name,
        },
    )

    print("[DONE] Monitoring metrics logged to MLflow.")
    return 0


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monitor last-month performance and drift (AUC, PSI, KS) via MLflow.")
    ap.add_argument("--mlflow-experiment", type=str, default="credit_risk_inference")
    ap.add_argument("--mlflow-artifact-name", type=str, default="predictions")
    ap.add_argument("--gold-labels-dir", type=str, default="datamart/gold/labels/")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--end-date", type=str, default=None)
    sys.exit(main(ap.parse_args()))
