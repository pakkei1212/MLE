# scripts/model_inference.py
import argparse, os
from datetime import datetime
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from utils.model_training_utils import spark_session, read_parquet_glob
import pyspark.sql.functions as F
from pyspark.sql.functions import col


def load_features_for_date(spark, features_dir: str, snapshotdate: str) -> pd.DataFrame:
    sdf = read_parquet_glob(spark, features_dir, "gold_pretrain_features")
    sdf = sdf.withColumn("label_snapshot_date", F.to_date(col("label_snapshot_date")))
    sdf = sdf.filter(col("label_snapshot_date") == F.lit(snapshotdate))
    return sdf.toPandas()


def resolve_model_uri(args):
    if args.model_uri:
        return args.model_uri, None
    return f"models:/{args.model_name}/{args.model_stage}", args.model_stage


def get_model_version_info(client: MlflowClient, model_uri: str, chosen_stage: str | None):
    try:
        if model_uri.startswith("models:/"):
            name = model_uri.split("/")[1]
            stage_or_ver = model_uri.split("/")[2]
            if stage_or_ver.isdigit():
                return int(stage_or_ver), chosen_stage or "None"
            for mv in client.search_model_versions(f"name = '{name}'"):
                if mv.current_stage == stage_or_ver:
                    return int(mv.version), mv.current_stage
        return None, chosen_stage or "None"
    except Exception:
        return None, chosen_stage or "None"


def main(args):
    # ---------- MLflow ----------
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_uri, chosen_stage = resolve_model_uri(args)

    # ---------- Load features ----------
    spark = spark_session()
    pdf = load_features_for_date(spark, args.gold_pretrain_features_dir, args.snapshotdate)
    spark.stop()
    if pdf.empty:
        raise RuntimeError(f"No features found in {args.gold_pretrain_features_dir} for snapshotdate={args.snapshotdate}")

    key_cols = [c for c in ["Customer_ID", "label_snapshot_date"] if c in pdf.columns]

    # ---------- Load model (sklearn flavor to use predict_proba) ----------
    import mlflow.sklearn as mls
    clf = mls.load_model(model_uri=model_uri)

    # ---------- Predict ----------
    proba = clf.predict_proba(pdf)
    if proba.ndim == 2:
        default_proba = proba[:, 1]
    else:
        # Rare fallback; normalize to [0,1]
        from sklearn.preprocessing import MinMaxScaler
        default_proba = MinMaxScaler().fit_transform(proba.reshape(-1, 1)).ravel()

    default_flag = (default_proba >= args.threshold).astype(int)

    # ---------- Build output DF ----------
    scored_at = datetime.now().astimezone().isoformat()
    out_df = pd.DataFrame(index=pdf.index)
    for k in key_cols:
        out_df[k] = pdf[k]
    out_df["default_proba"] = default_proba
    out_df["default"] = default_flag
    out_df["snapshotdate"] = args.snapshotdate
    out_df["scored_at"] = scored_at
    out_df["model_uri"] = model_uri

    mv, stage = get_model_version_info(client, model_uri, chosen_stage)
    out_df["model_version"] = mv if mv is not None else ""
    out_df["model_stage"] = stage

    # ---------- Persist (filesystem) ----------
    os.makedirs(args.predictions_out_dir, exist_ok=True)
    base = f"pred_{args.snapshotdate.replace('-', '')}"
    parquet_path = os.path.join(args.predictions_out_dir, f"{base}.parquet")
    csv_path = os.path.join(args.predictions_out_dir, f"{base}.csv") if args.write_csv else None

    out_df.to_parquet(parquet_path, index=False)
    if csv_path:
        out_df.to_csv(csv_path, index=False)

    print(f"[INFER] Wrote: {parquet_path}" + (f" and {csv_path}" if csv_path else ""))

    # ---------- MLflow (artifacts only) ----------
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"infer_{args.snapshotdate.replace('-', '_')}") as run:
        mlflow.set_tags({
            "snapshotdate": args.snapshotdate,
            "purpose": "inference",
            "model_uri": model_uri,
            "source": "airflow",
        })
        mlflow.log_params({
            "rows_scored": int(out_df.shape[0]),
            "threshold": float(args.threshold),
            "model_uri": model_uri,
            "model_version": mv if mv is not None else -1,
            "model_stage": stage,
        })
        mlflow.log_artifact(parquet_path, artifact_path="predictions")
        if csv_path:
            mlflow.log_artifact(csv_path, artifact_path="predictions")

        print(f"[INFER] run_id={run.info.run_id}")
        print(f"[INFER] artifact_uri: {mlflow.get_artifact_uri('predictions')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score Gold pretrain features and store predictions via MLflow (no metrics).")
    ap.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD to score")
    ap.add_argument("--gold-pretrain-features-dir", type=str,
                    default=os.path.join("datamart", "gold", "pretrain", "features") + "/")
    ap.add_argument("--predictions-out-dir", type=str,
                    default=os.path.join("datamart", "gold", "model_predictions") + "/")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for default flag")
    # MLflow model resolution
    ap.add_argument("--model-uri", default=None, help="Explicit URI (runs:/... or models:/name/Stage)")
    ap.add_argument("--model-name", default="credit_risk_model")
    ap.add_argument("--model-stage", default="Production")
    # Options
    ap.add_argument("--experiment", default="credit_risk_inference")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()
    main(args)
