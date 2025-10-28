# scripts/train_xgboost.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb
from ml_transforms import LoanTypeBinarizer

from utils.model_training_utils import (
    spark_session, read_parquet_glob,
    derive_windows, coerce_pandas_numeric,
    maybe_promote_to_production, write_model_registry_row,
    publish_simple_report, atomic_write_json
)
import pyspark.sql.functions as F
from pyspark.sql.functions import col


# ----------------------------
# helpers
# ----------------------------
def auc_any(est, X, y):
    if hasattr(est, "predict_proba"):
        s = est.predict_proba(X); s = s[:, 1] if s.ndim == 2 else s.ravel()
    elif hasattr(est, "decision_function"):
        s = est.decision_function(X)
    else:
        s = est.predict(X)
    return roc_auc_score(y, s)


# ----------------------------
# main
# ----------------------------
def main(args):
    # ---- time windows ----
    windows = derive_windows(args.model_train_date, args.train_test_months, args.oot_months)
    cfg = dict(
        model_train_date_str=args.model_train_date,
        train_test_period_months=args.train_test_months,
        oot_period_months=args.oot_months,
        train_test_ratio=args.train_ratio,
        random_state=args.random_state,
        n_iter=args.n_iter,
        **{k: str(v) for k, v in windows.items()}
    )
    print("\n=== CONFIG (XGBoost) ===")
    print(cfg)

    # ---- load spark data ----
    spark = spark_session()
    feat_sdf = read_parquet_glob(spark, args.gold_pretrain_features_dir, "gold_pretrain_features")
    lab_sdf  = read_parquet_glob(spark, args.gold_labels_dir, "gold_labels")

    feat_sdf = feat_sdf.withColumn("label_snapshot_date", F.to_date(col("label_snapshot_date")))
    lab_sdf  = lab_sdf.withColumn("snapshot_date", F.to_date(col("snapshot_date")))

    feat_sdf = feat_sdf.filter(
        (col("label_snapshot_date") >= F.lit(windows["train_test_start_date"].strftime("%Y-%m-%d"))) &
        (col("label_snapshot_date") <= F.lit(windows["oot_end_date"].strftime("%Y-%m-%d")))
    )
    lab_sdf = lab_sdf.filter(
        (col("snapshot_date") >= F.lit(windows["train_test_start_date"].strftime("%Y-%m-%d"))) &
        (col("snapshot_date") <= F.lit(windows["oot_end_date"].strftime("%Y-%m-%d")))
    )

    joined_sdf = (
        feat_sdf.alias("f")
        .join(
            lab_sdf.alias("l"),
            on=[col("f.Customer_ID") == col("l.Customer_ID"),
                col("f.label_snapshot_date") == col("l.snapshot_date")],
            how="inner"
        )
        .drop(col("l.Customer_ID"))
        .drop(col("l.snapshot_date"))
    )

    pdf = joined_sdf.toPandas()

    # ---- prep features/labels ----
    key_cols  = ["Customer_ID", "label_snapshot_date", "attributes_snapshot_date", "financials_snapshot_date"]
    label_col = "label"
    pdf = coerce_pandas_numeric(pdf, exclude_cols=set(key_cols + [label_col, "Age_bin", "Occupation", "Type_of_Loan"]))
    pdf["label_snapshot_date"] = pd.to_datetime(pdf["label_snapshot_date"]).dt.date

    mask_oot = (pdf["label_snapshot_date"] >= windows["oot_start_date"]) & (pdf["label_snapshot_date"] <= windows["oot_end_date"])
    mask_tt  = (pdf["label_snapshot_date"] >= windows["train_test_start_date"]) & (pdf["label_snapshot_date"] <= windows["train_test_end_date"])
    oot_pdf, tt_pdf = pdf.loc[mask_oot].copy(), pdf.loc[mask_tt].copy()

    cat_cols   = [c for c in ["Occupation", "Type_of_Loan", "Age_bin"] if c in pdf.columns]
    cat_single = [c for c in ["Occupation", "Age_bin"] if c in pdf.columns]
    has_loan   = "Type_of_Loan" in pdf.columns

    exclude = set(key_cols + [label_col] + cat_single + (["Type_of_Loan"] if has_loan else []))
    num_cols = [c for c in pdf.columns if c not in exclude and pd.api.types.is_numeric_dtype(pdf[c])]

    X_tt = tt_pdf[num_cols + cat_cols].copy()
    y_tt = tt_pdf[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tt, y_tt,
        test_size=(1 - args.train_ratio),
        random_state=args.random_state,
        shuffle=True,
        stratify=y_tt
    )
    X_oot = oot_pdf[num_cols + cat_cols].copy()
    y_oot = oot_pdf[label_col].astype(int)

    # ---- pipeline ----
    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_single),
    ]
    if has_loan:
        transformers.append(("loan_multi", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("multi", LoanTypeBinarizer(sep="|")),
        ]), ["Type_of_Loan"]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.random_state,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", xgb_clf)])

    param_dist = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__gamma": [0, 0.1],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_alpha": [0, 0.1, 1.0],
        "clf__reg_lambda": [1.0, 1.5, 2.0],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring=auc_any,
        cv=3,
        verbose=1,
        random_state=args.random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_

    # ---- eval ----
    def _eval(split_name, X, y):
        proba = best_pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        gini = 2 * auc - 1
        print(f"{split_name:>6} AUC={auc:.4f} | Gini={gini:.4f}")
        return auc, gini

    auc_train, gini_train = _eval("TRAIN", X_train, y_train)
    auc_test,  gini_test  = _eval(" TEST",  X_test,  y_test)
    auc_oot,   gini_oot   = _eval("  OOT",  X_oot,   y_oot)

    # ---- dumps ----
    model_version = f"credit_model_xgb_{args.model_train_date.replace('-', '_')}"
    outdir = os.path.join(args.model_bank_dir, model_version)
    os.makedirs(outdir, exist_ok=True)

    artefact = {
        "pipeline": best_pipe,
        "best_params": search.best_params_,
        "results": {
            "auc_train": float(auc_train), "gini_train": float(gini_train),
            "auc_test":  float(auc_test),  "gini_test":  float(gini_test),
            "auc_oot":   float(auc_oot),   "gini_oot":   float(gini_oot),
            "cv_best_auc": float(search.best_score_),
        },
        "data_stats": {
            "X_train_rows": int(X_train.shape[0]),
            "X_test_rows":  int(X_test.shape[0]),
            "X_oot_rows":   int(X_oot.shape[0]),
            "y_train_rate": float(y_train.mean()),
            "y_test_rate":  float(y_test.mean()),
            "y_oot_rate":   float(y_oot.mean()),
        },
        "config": cfg,
        "feature_columns": {
            "numeric": num_cols,
            "categorical": cat_cols,
        },
        "keys": ["Customer_ID", "label_snapshot_date"],
        "label_col": "label",
        "sources": {
            "features_dir": args.gold_pretrain_features_dir,
            "labels_dir": args.gold_labels_dir,
        },
        "flavor": "xgboost",
    }

    # Save artefact (cloudpickle)
    pkl_path = os.path.join(outdir, model_version + ".pkl")
    import cloudpickle as cp
    with open(pkl_path, "wb") as f:
        cp.dump(artefact, f)

    # Save metrics/artefact summaries
    atomic_write_json(os.path.join(outdir, "metrics.json"), artefact["results"])
    atomic_write_json(os.path.join(outdir, "artefact.json"), {k: v for k, v in artefact.items() if k != "pipeline"})

    # Simple HTML/PNG metrics report
    metrics = {"Train": float(auc_train), "Test": float(auc_test), "OOT": float(auc_oot)}
    publish_simple_report(outdir, model_version, metrics)

    # ---- registry (PARQUET) ----
    registry_dir = os.path.join("datamart", "gold", "model_registry")
    os.makedirs(registry_dir, exist_ok=True)

    promoted_flag = False
    promoted_at_iso = None

    # optional auto-promotion (compares OOT AUC vs current production)
    if args.auto_promote:
        promoted_flag = maybe_promote_to_production(
            artefact,
            version_str=model_version,
            production_dir=os.path.join(args.model_bank_dir, "production", "best"),
        )
        promoted_at_iso = datetime.now().astimezone().isoformat()

    write_model_registry_row(
        spark,
        registry_dir=registry_dir,
        model_version=model_version,
        train_start=windows["train_test_start_date"],
        train_end=windows["train_test_end_date"],
        oot_start=windows["oot_start_date"],
        oot_end=windows["oot_end_date"],
        auc_train=auc_train,
        auc_test=auc_test,
        auc_oot=auc_oot,
        promoted_flag=promoted_flag,
        promoted_at_iso=promoted_at_iso,
    )

    spark.stop()
    print("\n[DONE] XGBoost training completed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train XGBoost on Gold features + labels (Parquet registry)")
    ap.add_argument("--model-train-date", type=str, required=True)
    ap.add_argument("--train-test-months", type=int, default=12)
    ap.add_argument("--oot-months", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--random-state", type=int, default=88)
    ap.add_argument("--n-iter", type=int, default=50)

    ap.add_argument("--gold-pretrain-features-dir", type=str,
                    default=os.path.join("datamart", "gold", "pretrain", "features") + "/")
    ap.add_argument("--gold-labels-dir", type=str,
                    default=os.path.join("datamart", "gold", "labels") + "/")
    ap.add_argument("--model-bank-dir", type=str, default="model_bank")

    ap.add_argument("--auto-promote", action="store_true", help="Try to promote after training")

    args = ap.parse_args()
    main(args)
