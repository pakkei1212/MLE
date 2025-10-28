# scripts/model_training_utils.py
import os, io, json, glob, base64, tempfile
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import cloudpickle as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    DoubleType, BooleanType, TimestampType
)

# ----------------------------
# Spark + IO
# ----------------------------
def spark_session():
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("gold-pretrain-train")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def read_parquet_glob(spark, folder_path: str, label: str):
    files_list = [folder_path + os.path.basename(f)
                  for f in glob.glob(os.path.join(folder_path, '*'))]
    if not files_list:
        raise FileNotFoundError(f"[{label}] No parquet files found in: {folder_path}")
    df = spark.read.option("header", "true").parquet(*files_list)
    print(f"[LOAD] {label}: {len(files_list)} file(s) from {folder_path} rows={df.count()}")
    return df

def atomic_write_bytes(dst_path: str, data: bytes):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(dst_path), delete=False) as tmp:
        tmp.write(data)
        tmp_name = tmp.name
    os.replace(tmp_name, dst_path)

def atomic_write_json(dst_path: str, obj: dict):
    def _jsonify_default(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, set): return list(o)
        raise TypeError(f"Unserializable: {type(o)}")
    atomic_write_bytes(dst_path, json.dumps(obj, indent=2, default=_jsonify_default).encode("utf-8"))

def load_json_safe(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

# ----------------------------
# Dates / windows
# ----------------------------
def derive_windows(model_train_date_str: str, train_test_months: int, oot_months: int):
    model_train_date = datetime.strptime(model_train_date_str, "%Y-%m-%d").date()
    oot_end_date = model_train_date - timedelta(days=1)
    oot_start_date = model_train_date - relativedelta(months=oot_months)
    train_test_end_date = oot_start_date - timedelta(days=1)
    train_test_start_date = oot_start_date - relativedelta(months=train_test_months)
    return dict(
        model_train_date=model_train_date,
        oot_end_date=oot_end_date,
        oot_start_date=oot_start_date,
        train_test_end_date=train_test_end_date,
        train_test_start_date=train_test_start_date,
    )

# ----------------------------
# Pandas helpers
# ----------------------------
def coerce_pandas_numeric(pdf: pd.DataFrame, exclude_cols):
    for c in pdf.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(pdf[c]):
            continue
        pdf[c] = pd.to_numeric(pdf[c], errors="ignore")
    return pdf

# ----------------------------
# Registry
# ----------------------------
def write_model_registry_row(
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
    (sdf.write.mode("append").parquet(registry_dir))

# ----------------------------
# Promotion
# ----------------------------
def maybe_promote_to_production(
    artefact: dict,
    version_str: str,
    production_dir: str = "model_bank/production/best",
    epsilon: float = 1e-6
) -> bool:
    """Promote if OOT AUC is strictly better than current production by > epsilon."""
    os.makedirs(production_dir, exist_ok=True)

    new_auc = float(artefact.get("results", {}).get("auc_oot", float("nan")))
    prod_metrics_path = os.path.join(production_dir, "metrics.json")
    cur = load_json_safe(prod_metrics_path)
    cur_auc = float(cur.get("auc_oot", float("-inf")))

    if not np.isnan(new_auc) and (new_auc > cur_auc + epsilon):
        # Write model.pkl atomically
        pkl_bytes = cp.dumps(artefact)
        atomic_write_bytes(os.path.join(production_dir, "model.pkl"), pkl_bytes)
        # Write metrics + metadata
        atomic_write_json(os.path.join(production_dir, "metrics.json"), artefact["results"])
        art_sanitized = {k: v for k, v in artefact.items() if k != "pipeline"}
        atomic_write_json(os.path.join(production_dir, "artefact.json"), art_sanitized)
        atomic_write_bytes(os.path.join(production_dir, "model_version.txt"),
                           (version_str + "\n").encode("utf-8"))
        print(f"[PROMOTE] New production model: {version_str} (OOT AUC {new_auc:.4f} > {cur_auc:.4f})")
        return True

    print(f"[PROMOTE] Skip. New OOT AUC {new_auc:.4f} not better than production {cur_auc:.4f}.")
    return False

# ----------------------------
# Reporting
# ----------------------------
def publish_simple_report(outdir: str, model_version: str, metrics: dict):
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
    with open(html_path, "w") as f:
        f.write(html)

    atomic_write_json(os.path.join(outdir, "metrics.json"), metrics)
    print("\n[PUBLISH]")
    print(f"chart       : {png_path}")
    print(f"report      : {html_path}")
    print(f"metrics.json: {os.path.join(outdir, 'metrics.json')}")

# ----------------------------
# Common dataframe utilities
# ----------------------------
def attach_keys_first(df_features: pd.DataFrame, ref_df: pd.DataFrame,
                      keys=("Customer_ID","label_snapshot_date","attributes_snapshot_date","financials_snapshot_date")):
    keep = [k for k in keys if k in ref_df.columns]
    if not keep: return df_features
    attach = ref_df.loc[df_features.index, keep]
    return pd.concat([attach.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
