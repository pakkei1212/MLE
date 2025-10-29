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
        .appName("model-inference")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ----------------------------
# Robust artefact loader (cloudpickle + legacy redirect)
# ----------------------------
def _load_artefact(path: str):
    import pickle, io
    import cloudpickle

    scripts_dir = "/opt/airflow/scripts"
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        # If your transforms live here, keep this import (no-op if absent)
        from ml_transforms import LoanTypeBinarizer, NormalizeCategoricals  # noqa: F401
    except Exception:
        pass

    class RedirectUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name in {"LoanTypeBinarizer", "NormalizeCategoricals"}:
                module = "ml_transforms"
            return super().find_class(module, name)

    with open(path, "rb") as f:
        data = f.read()

    try:
        return cloudpickle.loads(data)
    except Exception:
        return RedirectUnpickler(io.BytesIO(data)).load()


def _load_production_model(model_bank_dir: str):
    prod_dir   = os.path.join(model_bank_dir, "production", "best")
    model_path = os.path.join(prod_dir, "model.pkl")
    ver_path   = os.path.join(prod_dir, "model_version.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No production model at {model_path}")

    artefact = _load_artefact(model_path)

    version = "production"
    try:
        with open(ver_path, "r") as f:
            v = f.read().strip()
            if v:
                version = v
    except Exception:
        pass

    print(f"[SELECT] Production model => {version} @ {model_path}")
    return version, model_path, artefact


# ----------------------------
# Metrics helper (optional if labels exist)
# ----------------------------
def _compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix
    )
    y_pred = (y_prob >= thr).astype(int)
    auc_roc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Confusion matrix (tn, fp, fn, tp)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "auc_roc": float(auc_roc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }


# ----------------------------
# Inference core
# ----------------------------
def main(
    snapshotdate: str,
    model_bank_dir: str,
    gold_pretrain_features_dir: str,
    predictions_out_dir: str,
    model_version: Optional[str] = None,
    use_production: bool = False,
    threshold: float = 0.5,
    gold_labels_dir: Optional[str] = None,   # optional, for metrics
):
    # Parse snapshot date
    snap_dt = datetime.strptime(snapshotdate, "%Y-%m-%d").date()

    # 1) Pick + load model artefact
    if use_production or not model_version:
        chosen_version, model_file, artefact = _load_production_model(model_bank_dir)
    else:
        model_file = os.path.join(model_bank_dir, model_version, model_version + ".pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Specified model not found: {model_file}")
        chosen_version = model_version
        artefact = _load_artefact(model_file)
        print(f"[SELECT] Using specified version: {chosen_version}")

    pipe = artefact["pipeline"]  # sklearn Pipeline (prep + clf)
    feature_columns = artefact.get("feature_columns", {})
    num_cols = feature_columns.get("numeric", []) or []
    cat_cols = feature_columns.get("categorical", []) or []
    feat_cols = list(dict.fromkeys(num_cols + cat_cols))

    # 2) Load Gold Pretrain features for snapshotdate
    spark = _spark()
    sdf = spark.read.option("header", "true").parquet(os.path.join(gold_pretrain_features_dir, "*"))
    sdf = sdf.withColumn("label_snapshot_date", to_date(col("label_snapshot_date")))
    sdf = sdf.filter(col("label_snapshot_date") == lit(snapshotdate))

    # Keep keys for output if they exist
    key_cols = [c for c in ["Customer_ID", "label_snapshot_date"] if c in sdf.columns]

    # Convert to pandas for sklearn
    cols_to_pull = key_cols + [c for c in feat_cols if c in sdf.columns]
    pdf = sdf.select(*cols_to_pull).toPandas()

    # Align columns: ensure every expected feature exists
    for c in feat_cols:
        if c not in pdf.columns:
            pdf[c] = np.nan  # let the pipeline's imputers handle it

    # Preserve the pipeline-trained order
    X_inf = pdf[feat_cols].copy()

    # 3) Predict
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_inf)
        score = proba[:, 1] if proba.ndim == 2 else np.ravel(proba)
    else:
        score = pipe.predict(X_inf)
        if score.ndim > 1:
            score = score.ravel()
        # If no proba available, treat prediction as score (0/1)
        score = score.astype(float)

    score = np.asarray(score, dtype=float)
    pred_flag = (score >= float(threshold)).astype(int)

    # 3b) Optional: load labels for metrics
    metrics = None
    if gold_labels_dir:
        try:
            ldf = spark.read.option("header", "true").parquet(os.path.join(gold_labels_dir, "*"))
            ldf = ldf.withColumn("snapshot_date", to_date(col("snapshot_date")))
            ldf = ldf.filter(col("snapshot_date") == lit(snapshotdate))
            lpdf = ldf.select("Customer_ID", "label").toPandas()
            merged = pd.merge(
                pdf[key_cols] if key_cols else pd.DataFrame({"__row_id__": np.arange(len(pdf))}),
                lpdf,
                how="left",
                on="Customer_ID" if "Customer_ID" in (key_cols or []) else None,
            )
            if "label" in merged.columns and merged["label"].notna().any():
                y_true = merged["label"].astype(float).round().astype(int).to_numpy()
                metrics = _compute_classification_metrics(y_true, score, float(threshold))
            else:
                print("[WARN] Labels not found for this snapshot; metrics will be omitted.")
        except Exception as e:
            print(f"[WARN] Unable to load labels for metrics: {e}")

    # 4) Build output frame
    out_pdf = pd.DataFrame({
        "Customer_ID": pdf["Customer_ID"].values if "Customer_ID" in pdf.columns else np.arange(len(score)),
        "snapshot_date": pd.to_datetime(pdf["label_snapshot_date"]).dt.date
            if "label_snapshot_date" in pdf.columns else snap_dt,
        "model_version": chosen_version,
        "predicted_default_risk": score.astype(float),
        "predicted_default": pred_flag.astype(int),
    })

    # 5) Write outputs (Parquet + summary JSON)
    out_dir = os.path.join(predictions_out_dir, chosen_version)
    os.makedirs(out_dir, exist_ok=True)

    parquet_name = f"{chosen_version}_predictions_{snapshotdate.replace('-', '_')}.parquet"
    out_parquet = os.path.join(out_dir, parquet_name)
    out_pdf.to_parquet(out_parquet, index=False)

    summary = {
        "model_version": chosen_version,
        "evaluation_time": snapshotdate,
        "sample_size": int(len(out_pdf)),
        "predicted_default_risk_min": float(np.nanmin(score)) if len(score) else None,
        "predicted_default_risk_max": float(np.nanmax(score)) if len(score) else None,
        "predicted_default_risk_mean": float(np.nanmean(score)) if len(score) else None,
        "threshold": float(threshold),
        "sources": {
            "model_file": model_file,
            "features_dir": gold_pretrain_features_dir,
        },
    }
    if metrics:
        summary.update({
            "auc_roc": metrics["auc_roc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"],
        })
    if gold_labels_dir:
        summary["sources"]["labels_dir"] = gold_labels_dir

    with open(os.path.join(out_dir, f"summary_{snapshotdate}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFER] model={chosen_version} rows_scored={len(out_pdf)}")
    print(f"[SAVED] {out_parquet}")

    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score predictions using a trained model (no MLflow).")
    ap.add_argument("--snapshotdate", required=True, type=str, help="YYYY-MM-DD")
    ap.add_argument("--model-bank-dir", default="model_bank/", type=str)
    ap.add_argument("--gold-pretrain-features-dir", default="datamart/pretrain_gold/features/", type=str)
    ap.add_argument("--predictions-out-dir", default="datamart/gold/model_predictions/", type=str)
    ap.add_argument("--model-version", default=None, type=str,
                    help="Specific version (e.g., credit_model_2024_01_01). If omitted, use production best.")
    ap.add_argument("--use-production", action="store_true",
                    help="Load model_bank/production/best/model.pkl (ignore --model-version).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for predicted_default (default 0.5).")
    ap.add_argument("--gold-labels-dir", type=str, default=None,
                    help="Optional: datamart/gold/labels/ to compute accuracy/precision/recall/F1 metrics.")

    args = ap.parse_args()

    main(
        snapshotdate=args.snapshotdate,
        model_bank_dir=args.model_bank_dir,
        gold_pretrain_features_dir=args.gold_pretrain_features_dir,
        predictions_out_dir=args.predictions_out_dir,
        model_version=args.model_version,
        use_production=args.use_production,
        threshold=args.threshold,
        gold_labels_dir=args.gold_labels_dir,
    )
