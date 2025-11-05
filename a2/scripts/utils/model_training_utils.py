# scripts/model_training_utils.py
import os, io, json, glob, base64, tempfile
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
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
    model_name: str = "credit_risk_model",
    epsilon: float = 1e-6,
) -> bool:
    """
    Promote to MLflow Production stage if the new OOT AUC is better than the
    current Production model by > epsilon. Operates fully inside MLflow.
    """
    client = MlflowClient()
    tracking_uri = mlflow.get_tracking_uri()
    print(f"[PROMOTE] Using MLflow tracking URI: {tracking_uri}")

    new_auc = float(artefact.get("results", {}).get("auc_oot", float("nan")))
    if np.isnan(new_auc):
        print("[PROMOTE] Skip. New model has invalid OOT AUC.")
        return False

    # --------------------------------------------------------
    # Get current Production model version and its auc_oot
    # --------------------------------------------------------
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_ver = prod_versions[0]
            prod_run_id = prod_ver.run_id
            cur_metrics = client.get_run(prod_run_id).data.metrics
            cur_auc = float(cur_metrics.get("auc_oot", float("-inf")))
            cur_ver_num = prod_ver.version
        else:
            cur_auc, cur_ver_num = float("-inf"), None
    except Exception as e:
        print(f"[PROMOTE] No current Production version found ({e}).")
        cur_auc, cur_ver_num = float("-inf"), None

    # --------------------------------------------------------
    # Compare and promote if better
    # --------------------------------------------------------
    if new_auc > cur_auc + epsilon:
        # Find target version in MLflow registry
        versions = client.search_model_versions(f"name='{model_name}'")
        target_ver = next(
            (v for v in versions if v.version == version_str or v.run_id == version_str), None
        )

        if not target_ver:
            print(f"[PROMOTE] Could not find model version '{version_str}' in MLflow registry.")
            return False

        client.transition_model_version_stage(
            name=model_name,
            version=target_ver.version,
            stage="Production",
            archive_existing_versions=True,
        )

        # Update the description for traceability
        desc = (
            f"Promoted on {datetime.now().isoformat()} | "
            f"auc_oot={new_auc:.4f} (prev={cur_auc:.4f})"
        )
        client.update_model_version(
            name=model_name,
            version=target_ver.version,
            description=desc,
        )

        print(
            f"[PROMOTE] ✅ Promoted version {target_ver.version} "
            f"to Production (AUC {new_auc:.4f} > {cur_auc:.4f})."
        )
        return True

    print(
        f"[PROMOTE] Skip. New OOT AUC {new_auc:.4f} "
        f"not better than current Production {cur_auc:.4f}."
    )
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
