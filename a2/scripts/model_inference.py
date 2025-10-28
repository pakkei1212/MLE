#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import io
import sys
import json
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import pyspark
from pyspark.sql.functions import col, to_date, lit


# ----------------------------
# Spark helper
# ----------------------------
def _spark():
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("model-inference-mlflow")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ----------------------------
# MLflow helpers
# ----------------------------
def _ensure_mlflow():
    try:
        import mlflow  # noqa
        from mlflow.tracking import MlflowClient  # noqa
    except Exception as e:
        raise RuntimeError(
            "MLflow is not available. Install mlflow and set MLFLOW_TRACKING_URI "
            "or pass --mlflow-tracking-uri."
        ) from e


def _mlflow_select_uri_by_train_date(
    tracking_uri: str,
    experiment: str,
    train_date: str
) -> Tuple[str, str]:
    """
    Returns (uri, info_string). Chooses best same-day run by metrics.auc_oot (DESC).
    URI points to 'runs:/{run_id}/model' so we load that exact artifact.
    """
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{experiment}' not found at {tracking_uri}.")

    flt = f'tags.train_date = "{train_date}"'
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=flt,
        order_by=["metrics.auc_oot DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No runs found for train_date={train_date} in experiment={experiment}.")

    best = runs.iloc[0]
    run_id = best.run_id
    auc = best.get("metrics.auc_oot", None)
    uri = f"runs:/{run_id}/model"
    info = f"[MLflow SELECT] best same-day run_id={run_id} auc_oot={auc} -> {uri}"
    return uri, info


def _mlflow_resolve_uri(
    tracking_uri: str,
    model_name: str,
    stage: Optional[str],
    version: Optional[str],
    experiment: Optional[str],
    train_date: Optional[str],
) -> Tuple[str, str, str]:
    """
    Resolve to a concrete MLflow model URI using precedence:
      1) explicit version -> models:/name/{version}
      2) stage (default 'Production') -> models:/name/{stage}
      3) train_date + experiment -> best by auc_oot for that date -> runs:/<id>/model
    Returns (chosen_label, uri, info_string)
    """
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)

    if version:
        uri = f"models:/{model_name}/{version}"
        return f"version_{version}", uri, f"[MLflow SELECT] explicit version -> {uri}"

    if stage:
        uri = f"models:/{model_name}/{stage}"
        return stage, uri, f"[MLflow SELECT] stage -> {uri}"

    if train_date and experiment:
        uri, info = _mlflow_select_uri_by_train_date(tracking_uri, experiment, train_date)
        return f"best_{train_date}", uri, info

    raise RuntimeError(
        "Cannot resolve MLflow model URI. Provide --mlflow-model-version, or --mlflow-stage, "
        "or both --train-date and --mlflow-experiment."
    )


def _mlflow_load_pyfunc(
    tracking_uri: str,
    model_name: str,
    stage: Optional[str],
    version: Optional[str],
    experiment: Optional[str],
    train_date: Optional[str],
) -> Tuple[str, str, "mlflow.pyfunc.PyFuncModel", List[str]]:
    """
    Returns (chosen_label, uri, pyfunc_model, feature_columns_from_signature)
    """
    _ensure_mlflow()
    import mlflow

    chosen_label, uri, info = _mlflow_resolve_uri(
        tracking_uri=tracking_uri,
        model_name=model_name,
        stage=stage,
        version=version,
        experiment=experiment,
        train_date=train_date,
    )
    print(info)

    model = mlflow.pyfunc.load_model(uri)

    # Try to fetch input schema / columns if logged with signature
    feature_columns = []
    try:
        sig = model.metadata.get_input_schema()
        if sig and sig.input_names():
            feature_columns = list(sig.input_names())
    except Exception:
        pass

    return chosen_label, uri, model, feature_columns


# ----------------------------
# Inference core
# ----------------------------
def main(
    snapshotdate: str,
    gold_pretrain_features_dir: str,
    predictions_out_dir: str,
    mlflow_tracking_uri: Optional[str],
    mlflow_model_name: str,
    mlflow_stage: Optional[str],
    mlflow_model_version: Optional[str],
    mlflow_experiment: Optional[str],
    train_date: Optional[str],
):
    # Parse snapshot date
    snap_dt = datetime.strptime(snapshotdate, "%Y-%m-%d").date()

    # Resolve MLflow tracking URI
    tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set. Pass --mlflow-tracking-uri or export the env var.")

    # 1) Pick + load model from MLflow
    chosen_label, model_uri, pyfunc_model, signature_cols = _mlflow_load_pyfunc(
        tracking_uri=tracking_uri,
        model_name=mlflow_model_name,
        stage=mlflow_stage,
        version=mlflow_model_version,
        experiment=mlflow_experiment,
        train_date=train_date,
    )
    print(f"[SELECT] Using MLflow model: {model_uri}")

    # 2) Load Gold Pretrain features for snapshotdate
    spark = _spark()
    sdf = spark.read.option("header", "true").parquet(os.path.join(gold_pretrain_features_dir, "*"))
    sdf = sdf.withColumn("label_snapshot_date", to_date(col("label_snapshot_date")))
    sdf = sdf.filter(col("label_snapshot_date") == lit(snapshotdate))

    # Keep keys for output if they exist
    key_cols = [c for c in ["Customer_ID", "label_snapshot_date"] if c in sdf.columns]

    # Columns to score: prefer signature (if provided), else all non-key columns
    if signature_cols:
        cols_needed = signature_cols
    else:
        cols_needed = [c for c in sdf.columns if c not in {"Customer_ID", "label_snapshot_date"}]

    cols_to_pull = list(dict.fromkeys(key_cols + [c for c in cols_needed if c in sdf.columns]))
    pdf = sdf.select(*cols_to_pull).toPandas()

    # Align columns: ensure every expected feature exists
    for c in cols_needed:
        if c not in pdf.columns:
            pdf[c] = np.nan

    # Preserve expected order
    X_inf = pdf[cols_needed].copy()

    # 3) Predict via pyfunc
    yhat = pyfunc_model.predict(X_inf)
    yhat = np.asarray(yhat)
    if yhat.ndim == 2 and yhat.shape[1] >= 2:
        score = yhat[:, -1]
    else:
        score = yhat.ravel()

    # 4) Build output frame
    out_pdf = pd.DataFrame({
        "Customer_ID": pdf["Customer_ID"].values if "Customer_ID" in pdf.columns else np.arange(len(score)),
        "snapshot_date": pd.to_datetime(pdf["label_snapshot_date"]).dt.date if "label_snapshot_date" in pdf.columns else snap_dt,
        "model_selector": chosen_label,   # e.g., "Production", "version_3", "best_2024-01-01"
        "model_uri": model_uri,
        "score": score.astype(float),
    })

    # 5) Write outputs (Parquet + summary JSON)
    out_dir = os.path.join(predictions_out_dir, chosen_label)
    os.makedirs(out_dir, exist_ok=True)

    parquet_name = f"{chosen_label}_predictions_{snapshotdate.replace('-', '_')}.parquet"
    out_parquet = os.path.join(out_dir, parquet_name)
    out_pdf.to_parquet(out_parquet, index=False)

    summary = {
        "model_selector": chosen_label,
        "model_uri": model_uri,
        "snapshotdate": snapshotdate,
        "n_scored": int(len(out_pdf)),
        "score_min": float(np.nanmin(score)) if len(score) else None,
        "score_max": float(np.nanmax(score)) if len(score) else None,
        "score_mean": float(np.nanmean(score)) if len(score) else None,
        "sources": {
            "features_dir": gold_pretrain_features_dir,
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_model_name": mlflow_model_name,
            "mlflow_stage_or_version": mlflow_model_version or mlflow_stage,
        },
    }
    if train_date:
        summary["sources"]["train_date"] = train_date

    with open(os.path.join(out_dir, f"summary_{snapshotdate}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFER] selector={chosen_label} rows_scored={len(out_pdf)}")
    print(f"[SAVED] {out_parquet}")

    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score predictions using a model from the MLflow Model Registry (pyfunc).")
    ap.add_argument("--snapshotdate", required=True, type=str, help="YYYY-MM-DD")
    ap.add_argument("--gold-pretrain-features-dir", default="datamart/pretrain_gold/features/", type=str)
    ap.add_argument("--predictions-out-dir", default="datamart/gold/model_predictions/", type=str)

    # MLflow selection
    ap.add_argument("--mlflow-tracking-uri", type=str, default=None,
                    help="Tracking server URI, e.g. http://mlflow:5000 (falls back to env MLFLOW_TRACKING_URI).")
    ap.add_argument("--mlflow-model-name", type=str, default="credit_risk_model",
                    help="Registered model name.")
    ap.add_argument("--mlflow-stage", type=str, default="Production",
                    help="Stage to load from (e.g., Production, Staging). Ignored if --mlflow-model-version is set.")
    ap.add_argument("--mlflow-model-version", type=str, default=None,
                    help="Explicit model version number to load from registry (e.g., '3').")
    ap.add_argument("--mlflow-experiment", type=str, default="credit_risk_training",
                    help="Experiment name used when searching by --train-date.")
    ap.add_argument("--train-date", type=str, default=None,
                    help="YYYY-MM-DD. If provided (with experiment), pick best same-day run by metrics.auc_oot.")

    args = ap.parse_args()

    main(
        snapshotdate=args.snapshotdate,
        gold_pretrain_features_dir=args.gold_pretrain_features_dir,
        predictions_out_dir=args.predictions_out_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_model_name=args.mlflow_model_name,
        mlflow_stage=args.mlflow_stage,
        mlflow_model_version=args.mlflow_model_version,
        mlflow_experiment=args.mlflow_experiment,
        train_date=args.train_date,
    )
