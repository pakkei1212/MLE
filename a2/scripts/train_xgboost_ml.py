# scripts/train_xgboost.py
import argparse, os
from datetime import datetime

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn  # enables sklearn pipeline logging
from mlflow.models import infer_signature
import mlflow.xgboost

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

from utils.model_training_utils import (
    spark_session, read_parquet_glob,
    derive_windows, coerce_pandas_numeric,
)
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def cls_metrics(y_true, y_proba, threshold=0.5, pos_label=1):
    """Return a flat dict of accuracy/precision/recall/f1 (micro/macro/weighted)."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def auc_any(est, X, y):
    if hasattr(est, "predict_proba"):
        s = est.predict_proba(X)
        s = s[:, 1] if s.ndim == 2 else s.ravel()
    elif hasattr(est, "decision_function"):
        s = est.decision_function(X)
    else:
        s = est.predict(X)
    return roc_auc_score(y, s)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(args):
    # Derive time windows
    windows = derive_windows(args.model_train_date, args.train_test_months, args.oot_months)
    cfg = dict(
        model_train_date_str=args.model_train_date,
        train_test_period_months=args.train_test_months,
        oot_period_months=args.oot_months,
        train_test_ratio=args.train_ratio,
        random_state=args.random_state,
        n_iter=args.n_iter,
        **{k: str(v) for k, v in windows.items()},
    )
    print("\n=== CONFIG (XGBoost) ===")
    print(cfg)

    # ---------- Load data from lake ----------
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

    # ---------- Prepare train/test/oot ----------
    key_cols  = ["Customer_ID", "label_snapshot_date", "attributes_snapshot_date", "financials_snapshot_date"]
    label_col = "label"

    pdf = coerce_pandas_numeric(
        pdf,
        exclude_cols=set(key_cols + [label_col, "Age_bin", "Occupation", "Type_of_Loan"])
    )
    pdf["label_snapshot_date"] = pd.to_datetime(pdf["label_snapshot_date"]).dt.date

    mask_oot = (pdf["label_snapshot_date"] >= windows["oot_start_date"]) & (
        pdf["label_snapshot_date"] <= windows["oot_end_date"]
    )
    mask_tt = (pdf["label_snapshot_date"] >= windows["train_test_start_date"]) & (
        pdf["label_snapshot_date"] <= windows["train_test_end_date"]
    )

    oot_pdf, tt_pdf = pdf.loc[mask_oot].copy(), pdf.loc[mask_tt].copy()

    cat_cols   = [c for c in ["Occupation", "Type_of_Loan", "Age_bin"] if c in pdf.columns]
    cat_single = [c for c in ["Occupation", "Age_bin"] if c in pdf.columns]
    has_loan   = "Type_of_Loan" in pdf.columns

    exclude = set(key_cols + [label_col] + cat_single + (["Type_of_Loan"] if has_loan else []))
    num_cols = [c for c in pdf.columns if c not in exclude and pd.api.types.is_numeric_dtype(pdf[c])]

    X_tt = tt_pdf[num_cols + cat_cols].copy()
    y_tt = tt_pdf[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tt,
        y_tt,
        test_size=(1 - args.train_ratio),
        random_state=args.random_state,
        shuffle=True,
        stratify=y_tt,
    )

    X_oot = oot_pdf[num_cols + cat_cols].copy()
    y_oot = oot_pdf[label_col].astype(int)

    # ---------- Preprocessing + model ----------
    transformers = [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat_single,
        ),
    ]
    # Note: if you had a custom "LoanTypeBinarizer", re-add here; otherwise
    # treat multi-label loan text upstream into columns before training.

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

    pipe = Pipeline([("prep", preprocessor), ("clf", xgb_clf)])

    # ---------- Hyperparam search ----------
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

    # ---------- Eval ----------
    def _eval(split_name, X, y, thr=0.5):
        proba = best_pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba); gini = 2*auc - 1
        m = cls_metrics(y, proba, threshold=thr)
        print(f"{split_name:>6} AUC={auc:.4f} | Gini={gini:.4f} | "
            f"Acc={m['accuracy']:.4f} | F1_w={m['f1_weighted']:.4f} | "
            f"P_w={m['precision_weighted']:.4f} | R_w={m['recall_weighted']:.4f}")
        return auc, gini, m


    auc_train, gini_train, m_train = _eval("TRAIN", X_train, y_train)
    auc_test,  gini_test,  m_test  = _eval(" TEST",  X_test,  y_test)
    auc_oot,   gini_oot,   m_oot   = _eval("  OOT",  X_oot,   y_oot)


    # ---------- MLflow logging (no local writes) ----------
    model_name = "credit_risk_model"        # MLflow Registry name
    train_date = args.model_train_date
    flavor = "xgboost"

    input_example = X_test.head(5)
    signature = infer_signature(input_example, best_pipe.predict_proba(input_example)[:, 1])

    mlflow.set_experiment("credit_risk_training")
    with mlflow.start_run(run_name=f"{flavor}_{train_date.replace('-','_')}"):
        # params & metrics
        mlflow.log_params(search.best_params_)
            
        mlflow.log_metrics({
            # AUC/Gini
            "auc_train": float(auc_train), "gini_train": float(gini_train),
            "auc_test":  float(auc_test),  "gini_test":  float(gini_test),
            "auc_oot":   float(auc_oot),   "gini_oot":   float(gini_oot),

            # TRAIN
            "accuracy_train":          float(m_train["accuracy"]),
            "precision_micro_train":   float(m_train["precision_micro"]),
            "precision_macro_train":   float(m_train["precision_macro"]),
            "precision_weighted_train":float(m_train["precision_weighted"]),
            "recall_micro_train":      float(m_train["recall_micro"]),
            "recall_macro_train":      float(m_train["recall_macro"]),
            "recall_weighted_train":   float(m_train["recall_weighted"]),
            "f1_micro_train":          float(m_train["f1_micro"]),
            "f1_macro_train":          float(m_train["f1_macro"]),
            "f1_weighted_train":       float(m_train["f1_weighted"]),

            # TEST
            "accuracy_test":           float(m_test["accuracy"]),
            "precision_micro_test":    float(m_test["precision_micro"]),
            "precision_macro_test":    float(m_test["precision_macro"]),
            "precision_weighted_test": float(m_test["precision_weighted"]),
            "recall_micro_test":       float(m_test["recall_micro"]),
            "recall_macro_test":       float(m_test["recall_macro"]),
            "recall_weighted_test":    float(m_test["recall_weighted"]),
            "f1_micro_test":           float(m_test["f1_micro"]),
            "f1_macro_test":           float(m_test["f1_macro"]),
            "f1_weighted_test":        float(m_test["f1_weighted"]),

            # OOT
            "accuracy_oot":            float(m_oot["accuracy"]),
            "precision_micro_oot":     float(m_oot["precision_micro"]),
            "precision_macro_oot":     float(m_oot["precision_macro"]),
            "precision_weighted_oot":  float(m_oot["precision_weighted"]),
            "recall_micro_oot":        float(m_oot["recall_micro"]),
            "recall_macro_oot":        float(m_oot["recall_macro"]),
            "recall_weighted_oot":     float(m_oot["recall_weighted"]),
            "f1_micro_oot":            float(m_oot["f1_micro"]),
            "f1_macro_oot":            float(m_oot["f1_macro"]),
            "f1_weighted_oot":         float(m_oot["f1_weighted"]),
        })

        # helpful tags
        mlflow.set_tags(
            {
                "train_date": train_date,
                "flavor": flavor,
                "source": "airflow",
            }
        )

        # log the sklearn pipeline (contains preprocessing + model)
        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            await_registration_for=0,
        )

        print(f"Artifact URI: {mlflow.get_artifact_uri('model')}")

    print(f"[MLflow] Logged under '{model_name}' (train_date={train_date}, flavor={flavor})")
    print("\n[DONE] XGBoost training completed.")

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train XGBoost on Gold features + labels (MLflow only)")
    ap.add_argument("--model-train-date", type=str, required=True)
    ap.add_argument("--train-test-months", type=int, default=12)
    ap.add_argument("--oot-months", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--random-state", type=int, default=88)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument(
        "--gold-pretrain-features-dir",
        type=str,
        default=os.path.join("datamart", "gold", "pretrain", "features") + "/",
    )
    ap.add_argument(
        "--gold-labels-dir",
        type=str,
        default=os.path.join("datamart", "gold", "labels") + "/",
    )
    args = ap.parse_args()
    main(args)
