# scripts/train_xgboost.py
import argparse, os
from datetime import datetime

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import mlflow.xgboost

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb
import optuna

from utils.model_training_utils import (
    spark_session, read_parquet_glob,
    derive_windows, coerce_pandas_numeric,
)
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def cls_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def auc_any(model, X, y):
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X)
        s = s[:, 1] if s.ndim == 2 else s.ravel()
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
    else:
        s = model.predict(X)
    return roc_auc_score(y, s)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(args):
    # ---------------- CONFIG ----------------
    windows = derive_windows(args.model_train_date, args.train_test_months, args.oot_months)
    print("\n=== CONFIG (XGBoost + Optuna) ===")
    print({**windows})

    # ---------------- LOAD DATA ----------------
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
            how="inner",
        )
        .drop(col("l.Customer_ID"))
        .drop(col("l.snapshot_date"))
    )

    pdf = joined_sdf.toPandas()
    spark.stop()

    # ---------------- SPLIT DATA ----------------
    key_cols = ["Customer_ID", "label_snapshot_date", "attributes_snapshot_date", "financials_snapshot_date"]
    label_col = "label"

    pdf = coerce_pandas_numeric(pdf, exclude_cols=set(key_cols + [label_col, "Age_bin", "Occupation", "Type_of_Loan"]))
    pdf["label_snapshot_date"] = pd.to_datetime(pdf["label_snapshot_date"]).dt.date

    mask_oot = (pdf["label_snapshot_date"] >= windows["oot_start_date"]) & (pdf["label_snapshot_date"] <= windows["oot_end_date"])
    mask_tt  = (pdf["label_snapshot_date"] >= windows["train_test_start_date"]) & (pdf["label_snapshot_date"] <= windows["train_test_end_date"])

    oot_pdf, tt_pdf = pdf.loc[mask_oot].copy(), pdf.loc[mask_tt].copy()

    cat_cols = [c for c in ["Occupation", "Type_of_Loan", "Age_bin"] if c in pdf.columns]
    cat_single = [c for c in ["Occupation", "Age_bin"] if c in pdf.columns]
    has_loan = "Type_of_Loan" in pdf.columns

    exclude = set(key_cols + [label_col] + cat_single + (["Type_of_Loan"] if has_loan else []))
    num_cols = [c for c in pdf.columns if c not in exclude and pd.api.types.is_numeric_dtype(pdf[c])]

    X_tt = tt_pdf[num_cols + cat_cols].copy()
    y_tt = tt_pdf[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tt, y_tt, test_size=(1 - args.train_ratio),
        random_state=args.random_state, stratify=y_tt
    )

    X_oot = oot_pdf[num_cols + cat_cols].copy()
    y_oot = oot_pdf[label_col].astype(int)

    # ---------------- PIPELINE ----------------
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_single)
    ])

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.3),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 2.0),
        }

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=args.random_state,
            **params
        )
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, y_pred))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", study_name="xgb_optuna")
    study.optimize(objective, n_trials=args.n_iter, show_progress_bar=True)
    best_params = study.best_params
    print("Best params:", best_params)

    best_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=args.random_state,
        **best_params
    )
    best_pipe = Pipeline([("prep", preprocessor), ("clf", best_model)])
    best_pipe.fit(X_train, y_train)

    # ---------------- EVAL ----------------
    def _eval(name, X, y):
        proba = best_pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        gini = 2 * auc - 1
        m = cls_metrics(y, proba)
        print(f"{name:>6}: AUC={auc:.4f} | Gini={gini:.4f} | F1_w={m['f1_weighted']:.4f}")
        return auc, gini, m

    auc_train, gini_train, m_train = _eval("TRAIN", X_train, y_train)
    auc_test,  gini_test,  m_test  = _eval(" TEST",  X_test,  y_test)
    auc_oot,   gini_oot,   m_oot   = _eval("  OOT",  X_oot,   y_oot)

    # ---------------- METADATA LOGGING ----------------
    def label_ratio(y):
        val, counts = np.unique(y, return_counts=True)
        return {f"label_ratio_{int(v)}": c / len(y) for v, c in zip(val, counts)}

    sample_info = {
        "sample_size_train": len(X_train),
        "sample_size_test": len(X_test),
        "sample_size_oot": len(X_oot),
        **{f"train_{k}": v for k, v in label_ratio(y_train).items()},
        **{f"test_{k}": v for k, v in label_ratio(y_test).items()},
        **{f"oot_{k}": v for k, v in label_ratio(y_oot).items()},
    }

    # ---------------- MLflow ----------------
    model_name = "credit_risk_model"
    flavor = "xgboost"
    train_date = args.model_train_date

    input_example = X_test.head(5)
    signature = infer_signature(input_example, best_pipe.predict_proba(input_example)[:, 1])

    mlflow.set_experiment("credit_risk_training")
    with mlflow.start_run(run_name=f"{flavor}_{train_date.replace('-','_')}"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "auc_train": auc_train, "gini_train": gini_train,
            "auc_test": auc_test, "gini_test": gini_test,
            "auc_oot": auc_oot, "gini_oot": gini_oot,
            **sample_info
        })
        mlflow.set_tags({
            "train_date": train_date,
            "flavor": flavor,
            "tuner": "optuna",
            "source": "airflow"
        })
        mlflow.sklearn.log_model(
            best_pipe, artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        print(f"[MLflow] Model logged. URI: {mlflow.get_artifact_uri('model')}")

    print("\n[DONE] XGBoost (Optuna) training completed.")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-train-date", type=str, required=True)
    ap.add_argument("--train-test-months", type=int, default=12)
    ap.add_argument("--oot-months", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--random-state", type=int, default=88)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--gold-pretrain-features-dir", type=str, default="datamart/gold/pretrain/features/")
    ap.add_argument("--gold-labels-dir", type=str, default="datamart/gold/labels/")
    args = ap.parse_args()
    main(args)
