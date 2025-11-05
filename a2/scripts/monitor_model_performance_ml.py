# monitor_model_performance_ml.py
import os, json, argparse, sys
import pandas as pd
from pathlib import Path
from datetime import date, datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# ---------- helpers ----------
def _summarize_predictions(df: pd.DataFrame) -> dict:
    s = pd.to_numeric(df["default_proba"], errors="coerce").dropna()
    return {
        "sample_size": int(len(s)),
        "default_proba_min": float(s.min()) if len(s) else None,
        "default_proba_max": float(s.max()) if len(s) else None,
        "default_proba_mean": float(s.mean()) if len(s) else None,
        "default_proba_std": float(s.std()) if len(s) else None,
    }

def _log_monitoring_to_mlflow(
    base_experiment: str,
    monitored_run_id: str,
    metrics: dict,
    summary: dict,
    last_month: date,
):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()

    # --- auto-create experiment if missing ---
    exp = mlflow.get_experiment_by_name(base_experiment)

    if exp is None:
        print(f"[MLFLOW] Experiment '{base_experiment}' not found. Creating it...")
        exp_id = client.create_experiment(name=base_experiment)
    else:
        exp_id = exp.experiment_id
    print(f"[MLFLOW] Using experiment '{base_experiment}' (id={exp_id}) for monitoring logs.")
    with mlflow.start_run(experiment_id=exp_id, run_name=f"monitor_{last_month}") as run:
        mlflow.set_tags({
            "monitoring_for_run_id": monitored_run_id,
            "monitor_month": last_month.isoformat(),
        })

        # Log metrics (AUC, accuracy, etc.) + summary stats
        mlflow.log_metrics({**metrics, **summary})

        # Log parameters (including last month explicitly)
        mlflow.log_params({
            "base_experiment": base_experiment,
            "source_run_id": monitored_run_id,
            "prediction_month": last_month.isoformat(),   # <-- logged as param
            "monitor_time": datetime.utcnow().isoformat(),
        })

        print(f"[MONITOR][MLFLOW] Logged monitoring metrics to run {run.info.run_id}")
    
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

def _load_labels_for_month(labels_dir: str, month: date) -> pd.DataFrame | None:
    root = Path(labels_dir)
    if not root.exists():
        print(f"[LABELS] Missing labels_dir: {labels_dir}")
        return None
    parts = []
    for p in root.glob("**/*.parquet"):
        try:
            df = pd.read_parquet(p)
            if "snapshot_date" not in df.columns or "label" not in df.columns:
                continue
            dd = _month_floor_series(df["snapshot_date"])
            df = df[dd == month][["Customer_ID","snapshot_date","label"]]
            if not df.empty:
                parts.append(df)
        except Exception:
            continue
    if not parts:
        print(f"[LABELS] No labels for month {month} under {labels_dir}")
        return None
    lab = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["Customer_ID","snapshot_date"], keep="last")
    lab["label"] = pd.to_numeric(lab["label"], errors="coerce").fillna(0).astype(int)
    lab["snapshot_date"] = pd.to_datetime(lab["snapshot_date"], errors="coerce").dt.date
    return lab

def _infer_month_from_tag_or_file(run, artifact_local_path: Path, month: date) -> bool:
    tag = (run.data.tags.get("snapshotdate") or run.data.tags.get("snapshot_date") or "").strip()
    if tag:
        try:
            snap_d = pd.to_datetime(tag).date()
            return _month_floor(snap_d) == month
        except Exception:
            pass
    try:
        df = pd.read_parquet(artifact_local_path)
        if "snapshot_date" in df.columns:
            dd = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
            return any(_month_floor(x) == month for x in dd.dropna())
    except Exception:
        pass
    return False

def _load_last_month_predictions_from_mlflow(
    experiment_name: str,
    artifact_name: str,   # can be a directory like "predictions" or a file pattern
    end_date: date,
) -> pd.DataFrame | None:
    """
    Find the newest FINISHED run in MLflow whose predictions artifact matches the LAST MONTH.
    - Supports artifact_name as a *directory* (e.g. "predictions") containing files like pred_YYYYMMDD.parquet
    - If artifact_name ends with ".parquet", we try that exact file (possibly in a subdir)
    - If multiple parquet files exist, we try to match the filename date to the target month
    Returns columns: ['Customer_ID','snapshot_date','default_proba'] (renamed if needed).
    """
    import re
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"[MLFLOW] Experiment '{experiment_name}' not found.")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=50,
    )
    if runs.empty:
        print("[MLFLOW] No FINISHED runs found.")
        return None

    target_month = _month_floor(end_date)
    target_token = f"{target_month.year}{target_month.month:02d}01"  # e.g. 20240501

    def _standardize_df(df: pd.DataFrame) -> pd.DataFrame | None:
        # score
        if "default_proba" not in df.columns:
            for k in ["score", "prob", "proba", "predicted_default_risk"]:
                if k in df.columns:
                    df = df.rename(columns={k: "default_proba"})
                    break
        # join keys / dates
        if "label_snapshot_date" in df.columns and "snapshot_date" not in df.columns:
            df = df.rename(columns={"label_snapshot_date": "snapshot_date"})
        need_keys = {"Customer_ID", "snapshot_date", "default_proba"}
        if not need_keys.issubset(df.columns):
            return None
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
        # filter to target month if possible
        mm = _month_floor_series(pd.Series(df["snapshot_date"]))
        df = df[mm == target_month][["Customer_ID", "snapshot_date", "default_proba"]]
        return df if not df.empty else None

    for _, r in runs.iterrows():
        run_id = r.run_id
        run = client.get_run(run_id)
        try:
            # List root artifacts
            root_entries = client.list_artifacts(run_id, path="")
            root_names = {e.path for e in root_entries}

            candidates: list[str] = []

            if artifact_name.endswith(".parquet"):
                # exact file or one-level nested match
                if artifact_name in root_names:
                    candidates = [artifact_name]
                else:
                    for e in root_entries:
                        if e.is_dir:
                            inner = client.list_artifacts(run_id, e.path)
                            for x in inner:
                                if x.path.endswith(artifact_name):
                                    candidates.append(x.path)
            else:
                # treat as a directory (e.g., "predictions")
                # collect all parquet files under it (one level)
                if artifact_name in root_names:
                    # grab parquet files inside
                    entries = client.list_artifacts(run_id, artifact_name)
                    for x in entries:
                        if x.path.endswith(".parquet"):
                            candidates.append(x.path)
                else:
                    # maybe it's nested one level deeper (rare)
                    for e in root_entries:
                        if e.is_dir:
                            inner = client.list_artifacts(run_id, e.path)
                            for x in inner:
                                if x.is_dir and x.path.endswith(artifact_name):
                                    deeper = client.list_artifacts(run_id, x.path)
                                    for y in deeper:
                                        if y.path.endswith(".parquet"):
                                            candidates.append(y.path)

            if not candidates:
                continue

            # Prefer a parquet matching pred_YYYYMM01.parquet
            exact = [c for c in candidates if re.search(rf"{target_token}\.parquet$", c)]
            ordered = exact + candidates  # exact match first, then any
            seen = set()
            ordered = [c for c in ordered if not (c in seen or seen.add(c))]

            for cand_path in ordered:
                local = Path(download_artifacts(artifact_uri=f"runs:/{run_id}/{cand_path}"))
                if not local.exists():
                    continue

                # If filename encodes a date, check match
                m = re.search(r"(\d{8})\.parquet$", local.name)  # pred_YYYYMMDD.parquet
                if m:
                    if m.group(1) != target_token:
                        # not the month we need; try next candidate
                        continue

                # Otherwise, fall back to checking content month
                df = pd.read_parquet(local)
                df = _standardize_df(df)
                if df is None or df.empty:
                    continue

                print(f"[MLFLOW] Loaded predictions from run {run_id} file {cand_path} for month {target_month}. Rows={len(df)}")
                return df.reset_index(drop=True)

        except Exception as e:
            print(f"[MLFLOW] Skip run {run_id}: {type(e).__name__}: {e}")
            continue

    print(f"[MLFLOW] No suitable predictions parquet found for month {target_month}.")
    return None

def _compute_binary_metrics(y_true, scores, thr: float = 0.5) -> dict:
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    y_true = pd.Series(y_true).astype(int).values
    s = pd.to_numeric(pd.Series(scores), errors="coerce").fillna(0).astype(float).values
    y_pred = (s >= thr).astype(int)

    out = {}
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, s))
    except Exception:
        out["auc_roc"] = None
    out["accuracy"]    = float(accuracy_score(y_true, y_pred))
    out["precision"]   = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]      = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1_macro"]    = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    return out

# ---------- main ----------
def main(args):
    end_date = _coerce_date(args.end_date)
    last_month = _month_floor(end_date)

    # MLflow last-month metrics (with label join)
    if args.use_mlflow:
        pred_df = _load_last_month_predictions_from_mlflow(
            experiment_name=args.mlflow_experiment,
            artifact_name=args.mlflow_artifact_name,
            end_date=end_date,
        )
        if pred_df is not None and len(pred_df):
            lab_df = _load_labels_for_month(args.gold_labels_dir, last_month)
            if lab_df is None or lab_df.empty:
                print("[MONITOR][MLFLOW-LAST-MONTH] No labels for last month; metrics skipped.")
            else:
                pred_df["snapshot_date"] = pd.to_datetime(pred_df["snapshot_date"], errors="coerce").dt.date
                lab_df["snapshot_date"]  = pd.to_datetime(lab_df["snapshot_date"],  errors="coerce").dt.date
                joined = pred_df.merge(
                    lab_df[["Customer_ID","snapshot_date","label"]],
                    on=["Customer_ID","snapshot_date"],
                    how="inner"
                )

                if joined.empty:
                    print("[MONITOR][MLFLOW-LAST-MONTH] No matches after join; metrics skipped.")
                else:
                    metrics = _compute_binary_metrics(joined["label"], joined["default_proba"], thr=args.threshold)
                    summary = _summarize_predictions(pred_df)

                    # Retrieve model info from run if possible
                    model_info = {
                        "mlflow_experiment": args.mlflow_experiment,
                        "mlflow_artifact_name": args.mlflow_artifact_name,
                    }

                    all_results = {
                        **metrics,
                        **summary,
                        **model_info,
                    }

                    #print("[MONITOR][MLFLOW-LAST-MONTH] metrics:", json.dumps(all_results, indent=2))

                    out_root = Path(args.out_dir)
                    out_root.mkdir(parents=True, exist_ok=True)
                    with open(out_root / f"metrics_last_month_{last_month.isoformat()}.json", "w") as f:
                        json.dump(all_results, f, indent=2)

                    # Log evaluation results to MLflow
                    # find the run_id from which predictions came
                    # (we can modify _load_last_month_predictions_from_mlflow to return it)
                    if hasattr(pred_df, "_mlflow_run_id"):
                        run_id = getattr(pred_df, "_mlflow_run_id")
                    else:
                        run_id = "unknown"

                    _log_monitoring_to_mlflow(
                        base_experiment=args.mlflow_experiment,
                        monitored_run_id=run_id,
                        metrics=metrics,
                        summary=summary,
                        last_month=last_month,
                    )

        else:
            print("[MONITOR][MLFLOW-LAST-MONTH] No predictions DF; skipping MLflow metrics.")

    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monitor last-month metrics by reading predictions from MLflow and joining Gold labels.")
    ap.add_argument("--use-mlflow", action="store_true")
    ap.add_argument("--mlflow-experiment", type=str, default="credit_risk_inference")
    ap.add_argument("--mlflow-artifact-name", type=str, default="predictions")
    ap.add_argument("--gold-labels-dir", type=str, default="datamart/gold/labels/")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--out-dir", type=str, default="datamart/gold/model_monitoring/")
    sys.exit(main(ap.parse_args()))
