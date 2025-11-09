# scripts/model_training.py
import argparse
import os
import glob
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json, tempfile, cloudpickle as cp
import xgboost as xgb
from ml_transforms import LoanTypeBinarizer
import io, os, json, base64, hashlib, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, BooleanType, TimestampType
import pandas as pd
from datetime import datetime


# ----------------------------
# Helpers
# ----------------------------
def _spark():
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("gold-pretrain-train")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def _write_model_registry_row(
    spark,
    registry_dir: str,
    model_version: str,
    train_start, train_end,
    oot_start, oot_end,
    auc_train: float, auc_test: float, auc_oot: float,
    promoted_flag: bool,
    promoted_at_iso: str | None
):
    rows = [{
        "model_version": model_version,
        "train_start": pd.to_datetime(train_start).date(),
        "train_end":   pd.to_datetime(train_end).date(),
        "oot_start":   pd.to_datetime(oot_start).date(),
        "oot_end":     pd.to_datetime(oot_end).date(),
        "auc_train": float(auc_train),
        "auc_test":  float(auc_test),
        "auc_oot":   float(auc_oot),
        "promoted_flag": bool(promoted_flag),
        "promoted_at": None if promoted_at_iso is None else pd.to_datetime(promoted_at_iso),
    }]
    pdf = pd.DataFrame(rows)

    schema = StructType([
        StructField("model_version", StringType(), False),
        StructField("train_start", DateType(), True),
        StructField("train_end",   DateType(), True),
        StructField("oot_start",   DateType(), True),
        StructField("oot_end",     DateType(), True),
        StructField("auc_train",   DoubleType(), True),
        StructField("auc_test",    DoubleType(), True),
        StructField("auc_oot",     DoubleType(), True),
        StructField("promoted_flag", BooleanType(), True),
        StructField("promoted_at",  TimestampType(), True),
    ])
    sdf = spark.createDataFrame(pdf, schema=schema)
    (sdf.write
        .mode("append")
        .parquet(registry_dir))
    
def _atomic_write_bytes(dst_path: str, data: bytes):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(dst_path), delete=False) as tmp:
        tmp.write(data)
        tmp_name = tmp.name
    os.replace(tmp_name, dst_path)

def _atomic_write_json(dst_path: str, obj: dict):
    _atomic_write_bytes(dst_path, json.dumps(obj, indent=2).encode("utf-8"))

def _load_json_safe(path: str) -> dict:
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception:
        return {}

def maybe_promote_to_production(artefact: dict,
                                version_str: str,
                                production_dir: str = "model_bank/production/best",
                                epsilon: float = 1e-6) -> bool:
    """Promote if OOT AUC is strictly better (by > epsilon) than current production."""
    os.makedirs(production_dir, exist_ok=True)

    new_auc = float(artefact.get("results", {}).get("auc_oot", float("nan")))
    prod_metrics_path = os.path.join(production_dir, "metrics.json")
    cur = _load_json_safe(prod_metrics_path)
    cur_auc = float(cur.get("auc_oot", float("-inf")))

    if not np.isnan(new_auc) and (new_auc > cur_auc + epsilon):
        # Write model.pkl atomically
        pkl_bytes = cp.dumps(artefact)
        _atomic_write_bytes(os.path.join(production_dir, "model.pkl"), pkl_bytes)

        # Write metrics + metadata
        _atomic_write_json(os.path.join(production_dir, "metrics.json"), artefact["results"])
        # Optional: sanitized artefact (no pipeline) for inspection
        art_sanitized = {k: v for k, v in artefact.items() if k != "pipeline"}
        _atomic_write_json(os.path.join(production_dir, "artefact.json"), art_sanitized)
        _atomic_write_bytes(os.path.join(production_dir, "model_version.txt"),
                            (version_str + "\n").encode("utf-8"))

        print(f"[PROMOTE] New production model: {version_str} (OOT AUC {new_auc:.4f} > {cur_auc:.4f})")
        return True

    print(f"[PROMOTE] Skip. New OOT AUC {new_auc:.4f} not better than production {cur_auc:.4f}.")
    return False

def _read_parquet_glob(spark, folder_path: str, label: str):
    files_list = [folder_path + os.path.basename(f)
                  for f in glob.glob(os.path.join(folder_path, '*'))]
    if not files_list:
        raise FileNotFoundError(f"[{label}] No parquet files found in: {folder_path}")
    df = spark.read.option("header", "true").parquet(*files_list)
    print(f"[LOAD] {label}: {len(files_list)} file(s) from {folder_path} rows={df.count()}")
    return df


def _coerce_pandas_numeric(pdf: pd.DataFrame, exclude_cols):
    for c in pdf.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(pdf[c]):
            continue
        # try to coerce numeric columns that came as object
        pdf[c] = pd.to_numeric(pdf[c], errors="ignore")
    return pdf

    
# ----------------------------
# Main training
# ----------------------------
def main(
    model_train_date_str: str,
    train_test_period_months: int,
    oot_period_months: int,
    train_test_ratio: float,
    random_state: int,
    n_iter: int,
    gold_pretrain_features_dir: str,
    gold_labels_dir: str,
    model_bank_dir: str,
):
    # -------- dates & windows --------
    model_train_date = datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
    oot_end_date = model_train_date - timedelta(days=1)
    oot_start_date = model_train_date - relativedelta(months=oot_period_months)
    train_test_end_date = oot_start_date - timedelta(days=1)
    train_test_start_date = oot_start_date - relativedelta(months=train_test_period_months)

    config = {
        "model_train_date_str": model_train_date_str,
        "train_test_period_months": train_test_period_months,
        "oot_period_months": oot_period_months,
        "train_test_ratio": train_test_ratio,
        "model_train_date": str(model_train_date),
        "oot_end_date": str(oot_end_date),
        "oot_start_date": str(oot_start_date),
        "train_test_end_date": str(train_test_end_date),
        "train_test_start_date": str(train_test_start_date),
        "random_state": random_state,
        "n_iter": n_iter,
    }
    print("\n=== CONFIG ===")
    pprint.pprint(config)

    # -------- Spark load --------
    spark = _spark()

    # pretrain features (gold)
    feat_sdf = _read_parquet_glob(spark, gold_pretrain_features_dir, label="gold_pretrain_features")
    # labels (gold)
    lab_sdf = _read_parquet_glob(spark, gold_labels_dir, label="gold_labels")

    # Ensure date types
    # - pretrain has label_snapshot_date
    # - labels has snapshot_date
    feat_sdf = feat_sdf.withColumn("label_snapshot_date", F.to_date(col("label_snapshot_date")))
    lab_sdf = lab_sdf.withColumn("snapshot_date", F.to_date(col("snapshot_date")))

    # Restrict to the full window [train_test_start_date, oot_end_date]
    feat_sdf = feat_sdf.filter(
        (col("label_snapshot_date") >= F.lit(train_test_start_date.strftime("%Y-%m-%d"))) &
        (col("label_snapshot_date") <= F.lit(oot_end_date.strftime("%Y-%m-%d")))
    )
    lab_sdf = lab_sdf.filter(
        (col("snapshot_date") >= F.lit(train_test_start_date.strftime("%Y-%m-%d"))) &
        (col("snapshot_date") <= F.lit(oot_end_date.strftime("%Y-%m-%d")))
    )

    print("[FILTER] features rows in window:", feat_sdf.count())
    print("[FILTER] labels rows in window:", lab_sdf.count())

    # Join keys: Customer_ID and equality of dates (label_snapshot_date == snapshot_date)
    joined_sdf = (
        feat_sdf.alias("f")
        .join(lab_sdf.alias("l"),
              on=[col("f.Customer_ID") == col("l.Customer_ID"),
                  col("f.label_snapshot_date") == col("l.snapshot_date")],
              how="inner")
        .drop(col("l.Customer_ID"))
        .drop(col("l.snapshot_date"))
    )

    print("[JOIN] rows:", joined_sdf.count())

    # Pull to pandas for sklearn
    pdf = joined_sdf.toPandas()

    # Coerce numerics (Spark->Pandas can keep numeric as object if nulls/strings appeared)
    key_cols = ["Customer_ID", "label_snapshot_date", "attributes_snapshot_date", "financials_snapshot_date"]
    label_col = "label"
    pdf = _coerce_pandas_numeric(pdf, exclude_cols=set(key_cols + [label_col, "Age_bin", "Occupation", "Type_of_Loan"]))

    # Split by time windows using the label date (we kept both, but label_snapshot_date matches the label row)
    pdf["label_snapshot_date"] = pd.to_datetime(pdf["label_snapshot_date"]).dt.date

    mask_oot = (pdf["label_snapshot_date"] >= oot_start_date) & (pdf["label_snapshot_date"] <= oot_end_date)
    mask_tt  = (pdf["label_snapshot_date"] >= train_test_start_date) & (pdf["label_snapshot_date"] <= train_test_end_date)

    oot_pdf = pdf.loc[mask_oot].copy()
    tt_pdf  = pdf.loc[mask_tt].copy()

    # ----------------------------
    # Features/Targets
    # ----------------------------
    # Categorical features
    cat_cols = [c for c in ["Occupation", "Type_of_Loan", "Age_bin"] if c in pdf.columns]

    # NOTE on Age_bin:
    # We choose ONE-HOT encoding by default to preserve non-linear effects across bins.
    # If you prefer an ordinal mapping (e.g., under_18<18_24<25_34<...),
    # swap OneHotEncoder for an OrdinalEncoder with a fixed category order.

    # Numeric features = all non-key, non-label, non-categorical
    exclude = set(key_cols + [label_col] + cat_cols)
    num_cols = [c for c in pdf.columns if c not in exclude and pd.api.types.is_numeric_dtype(pdf[c])]

    # Guard: if Type_of_Loan is multi-valued (e.g., "Auto, Credit Card"), consider splitting upstream.
    # Here we treat it as a single categorical token as-is.

    # Train/Test split from the time-windowed tt_pdf
    X_tt = tt_pdf[num_cols + cat_cols].copy()
    y_tt = tt_pdf[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tt, y_tt,
        test_size=(1 - train_test_ratio),
        random_state=random_state,
        shuffle=True,
        stratify=y_tt
    )

    # OOT
    X_oot = oot_pdf[num_cols + cat_cols].copy()
    y_oot = oot_pdf[label_col].astype(int)

    print(f"[SIZES] X_train={X_train.shape}, X_test={X_test.shape}, X_oot={X_oot.shape}")
    print(f"[LABELS] y_train mean={y_train.mean():.3f}  y_test mean={y_test.mean():.3f}  y_oot mean={y_oot.mean():.3f}")

    # ----------------------------
    # Preprocessing + Model pipeline
    # ----------------------------
    # Categorical features (single-valued OHE)
    cat_single = [c for c in ["Occupation", "Age_bin"] if c in pdf.columns]

    # Multi-valued categorical handled by custom binarizer
    has_loan = "Type_of_Loan" in pdf.columns

    # Numeric features = all non-key, non-label, non-categorical
    exclude = set(key_cols + [label_col] + cat_single + (["Type_of_Loan"] if has_loan else []))
    num_cols = [c for c in pdf.columns if c not in exclude and pd.api.types.is_numeric_dtype(pdf[c])]

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

    # HARD CHECKS
    uniq = set(y_tt.unique()).union(set(y_oot.unique()))
    assert uniq.issubset({0, 1}), f"Labels must be binary 0/1, got {sorted(uniq)}"

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
        ,  # fast on CPU
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", xgb_clf),
    ])

    # Hyperparameter search space (prefixed with 'clf__' for pipeline)
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

    #auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    def auc_any(est, X, y):
        # Prefer predict_proba; fall back to decision_function; then predict
        if hasattr(est, "predict_proba"):
            s = est.predict_proba(X)
            s = s[:, 1] if s.ndim == 2 else s.ravel()
        elif hasattr(est, "decision_function"):
            s = est.decision_function(X)
        else:
            s = est.predict(X)
        return roc_auc_score(y, s)

    #auc_scorer = make_scorer(_auc_any, greater_is_better=True)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=auc_any,
        cv=3,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        refit=True,  # keep best pipeline fitted
    )

    # Fit
    search.fit(X_train, y_train)

    print("\n[SEARCH] Best params:", search.best_params_)
    print("[SEARCH] Best CV AUC:", f"{search.best_score_:.4f}")

    best_pipe = search.best_estimator_

    # After: best_pipe = search.best_estimator_

    # --- Dump post-preprocessing feature matrices ---
    model_version = f"credit_model_{model_train_date_str.replace('-','_')}" 
    dump_dir = os.path.join(model_bank_dir, model_version) 
    os.makedirs(dump_dir, exist_ok=True)

    preproc = best_pipe.named_steps["prep"]  # fitted ColumnTransformer
    feat_names = preproc.get_feature_names_out()

    def _with_keys(df_features: pd.DataFrame, ref_df: pd.DataFrame, keys=("Customer_ID","label_snapshot_date","attributes_snapshot_date","financials_snapshot_date")):
        keep = [k for k in keys if k in ref_df.columns]
        if not keep:
            return df_features
        attach = ref_df.loc[df_features.index, keep]
        # Put keys first for readability
        return pd.concat([attach.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)

    # Transform each split using the fitted preprocessor
    X_train_tr = preproc.transform(X_train)
    X_test_tr  = preproc.transform(X_test)
    X_oot_tr   = preproc.transform(X_oot)

    # Wrap into DataFrames with feature names
    X_train_pp = pd.DataFrame(X_train_tr, columns=feat_names, index=X_train.index)
    X_test_pp  = pd.DataFrame(X_test_tr,  columns=feat_names, index=X_test.index)
    X_oot_pp   = pd.DataFrame(X_oot_tr,   columns=feat_names, index=X_oot.index)

    # Attach keys/dates from original split frames for sanity checks
    X_train_pp = _with_keys(X_train_pp, tt_pdf)
    X_test_pp  = _with_keys(X_test_pp,  tt_pdf)
    X_oot_pp   = _with_keys(X_oot_pp,   oot_pdf)

    # Save to CSV next to the raw-split dumps you already write
    X_train_pp.to_csv(os.path.join(dump_dir, "X_train_preprocessed.csv"), index=False)
    X_test_pp.to_csv(os.path.join(dump_dir, "X_test_preprocessed.csv"), index=False)
    X_oot_pp.to_csv(os.path.join(dump_dir, "X_oot_preprocessed.csv"), index=False)

    print(f"[DUMPS] Preprocessed CSVs saved under: {dump_dir}")

    # Evaluate
    def _eval(split_name, X, y):
        proba = best_pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        gini = 2 * auc - 1
        print(f"{split_name:>6} AUC={auc:.4f} | Gini={gini:.4f}")
        return auc, gini

    auc_train, gini_train = _eval("TRAIN", X_train, y_train)
    auc_test,  gini_test  = _eval(" TEST", X_test, y_test)
    auc_oot,   gini_oot   = _eval("  OOT", X_oot, y_oot)

    # ----------------------------
    # Save artefact
    # ----------------------------
    artefact = {
        "pipeline": best_pipe,  # includes preprocessing + xgb model
        "best_params": search.best_params_,
        "results": {
            "auc_train": auc_train, "gini_train": gini_train,
            "auc_test": auc_test,   "gini_test": gini_test,
            "auc_oot": auc_oot,     "gini_oot": gini_oot,
            "cv_best_auc": float(search.best_score_),
        },
        "data_stats": {
            "X_train_rows": int(X_train.shape[0]),
            "X_test_rows": int(X_test.shape[0]),
            "X_oot_rows": int(X_oot.shape[0]),
            "y_train_rate": float(y_train.mean()),
            "y_test_rate": float(y_test.mean()),
            "y_oot_rate": float(y_oot.mean()),
        },
        "config": config,
        "feature_columns": {
            "numeric": num_cols,
            "categorical": cat_cols,
        },
        "keys": ["Customer_ID", "label_snapshot_date"],
        "label_col": label_col,
        "sources": {
            "features_dir": gold_pretrain_features_dir,
            "labels_dir": gold_labels_dir,
        },
    }

    model_version = f"credit_model_{model_train_date_str.replace('-','_')}"
    outdir = os.path.join(model_bank_dir, model_version)
    os.makedirs(outdir, exist_ok=True)
    pkl_path = os.path.join(outdir, model_version + ".pkl")

    print("\n[MODEL SAVE]")
    print(f"model_version : {model_version}")
    print(f"outdir        : {outdir}")
    print(f"pkl_path      : {pkl_path}")

    with open(pkl_path, "wb") as f:
        cp.dump(artefact, f)

    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(artefact["results"], f, indent=2)

    print("[METRICS SAVE]")
    print(f"metrics_path  : {metrics_path}")
    print(f"AUC train/test/oot : {artefact['results'].get('auc_train'):.4f} / "
      f"{artefact['results'].get('auc_test'):.4f} / {artefact['results'].get('auc_oot'):.4f}")

    # ---- attempt auto-promotion to production/best ----
    prod_dir = os.path.join(model_bank_dir, "production", "best")
    print("\n[PROMOTION]")
    print(f"production_dir: {prod_dir}")
    promoted = maybe_promote_to_production(
        artefact,
        version_str=model_version,
        production_dir=prod_dir
    )
    print(f"promoted      : {promoted}")

    '''# ... after you compute auc_train, auc_test, auc_oot and save artefact/promotion ...
    registry_dir = os.path.join("datamart", "gold", "model_registry")
    os.makedirs(registry_dir, exist_ok=True)

    # If you already have promotion logic, set these:
    promoted_flag = artefact.get("promotion", {}).get("promoted", False)
    promoted_at_iso = artefact.get("promotion", {}).get("promoted_at")  # e.g. "2025-10-28T10:35:00+08:00"

    _write_model_registry_row(
        spark,
        registry_dir=registry_dir,
        model_version=model_version,
        train_start=config["train_test_start_date"],
        train_end=config["train_test_end_date"],
        oot_start=config["oot_start_date"],
        oot_end=config["oot_end_date"],
        auc_train=auc_train,
        auc_test=auc_test,
        auc_oot=auc_oot,
        promoted_flag=promoted_flag,
        promoted_at_iso=promoted_at_iso
    )
    print(f"[REGISTRY] appended row for {model_version} -> {registry_dir}")'''

    # ---- done with Spark ----
    spark.stop()
    print("\n[DONE] Training pipeline completed.")

    # ==== PUBLISH METRICS & ARTEFACTS (inline, inside training) ====
    def _jsonify_default(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, set): return list(o)
        raise TypeError(f"Unserializable: {type(o)}")

    def _ensure_dir(p):
        os.makedirs(p, exist_ok=True); return p

    def _atomic_write_bytes(dst_path: str, data: bytes):
        _ensure_dir(os.path.dirname(dst_path))
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(dst_path), delete=False) as tmp:
            tmp.write(data)
            name = tmp.name
        os.replace(name, dst_path)

    def _atomic_write_json(dst_path: str, obj: dict):
        _atomic_write_bytes(dst_path, json.dumps(obj, indent=2, default=_jsonify_default).encode("utf-8"))

    # --- Where to publish ---
    metrics = {
        "Train": float(artefact["results"]["auc_train"]),
        "Test":  float(artefact["results"]["auc_test"]),
        "OOT":   float(artefact["results"]["auc_oot"]),
    }

    # --- Line chart (Train/Test/OOT) ---
    xs = ["Train", "Test", "OOT"]
    ys = [metrics[x] for x in xs]

    fig = plt.figure(figsize=(5.0, 3.2))
    ax = plt.gca()
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title(model_version)
    plt.tight_layout()

    png_path = os.path.join(outdir, "auc_line.png")
    fig.savefig(png_path, bbox_inches="tight")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    html = f"""<html><body>
    <h3>{model_version}</h3>
    <img src="data:image/png;base64,{img_b64}" />
    <pre>{json.dumps(metrics, indent=2)}</pre>
    </body></html>"""
    html_path = os.path.join(outdir, "report.html")
    with open(html_path, "w") as f: f.write(html)

    # --- Save JSONs (metrics + sanitized artefact + sections) ---
    _atomic_write_json(os.path.join(outdir, "metrics.json"), metrics)

    art_sanitized = {k: v for k, v in artefact.items() if k != "pipeline"}
    _atomic_write_json(os.path.join(outdir, "artefact.json"), art_sanitized)

    for k, v in art_sanitized.items():
        if isinstance(v, dict):
            _atomic_write_json(os.path.join(outdir, f"{k}.json"), v)

    print("\n[PUBLISH]")
    print(f"chart       : {png_path}")
    print(f"report      : {html_path}")
    print(f"metrics.json: {os.path.join(outdir, 'metrics.json')}")
    print(f"artefact.json (no pipeline): {os.path.join(outdir, 'artefact.json')}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train model using Gold Pretrain features + Gold labels")
    p.add_argument("--model-train-date", type=str, required=True, help="YYYY-MM-DD (train window anchor)")
    p.add_argument("--train-test-months", type=int, default=12, help="Months in train+test window (backwards from OOT start)")
    p.add_argument("--oot-months", type=int, default=2, help="Months in OOT window (immediately before anchor date)")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train share within train+test window")
    p.add_argument("--random-state", type=int, default=88)
    p.add_argument("--n-iter", type=int, default=50, help="RandomizedSearch iterations")

    # paths
    p.add_argument("--gold-pretrain-features-dir", type=str,
                   default=os.path.join("datamart", "gold", "pretrain", "features") + "/")
    p.add_argument("--gold-labels-dir", type=str,
                   default=os.path.join("datamart", "gold", "labels") + "/")
    p.add_argument("--model-bank-dir", type=str, default="model_bank/")

    args = p.parse_args()

    main(
        model_train_date_str=args.model_train_date,
        train_test_period_months=args.train_test_months,
        oot_period_months=args.oot_months,
        train_test_ratio=args.train_ratio,
        random_state=args.random_state,
        n_iter=args.n_iter,
        gold_pretrain_features_dir=args.gold_pretrain_features_dir,
        gold_labels_dir=args.gold_labels_dir,
        model_bank_dir=args.model_bank_dir,
    )
