# utils/data_processing_pretrain_gold_table.py

import os
import os, glob
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

# ------------------------------
# helpers
# ------------------------------
def _cat_path(dir_path: str, filename: str) -> str:
    return os.path.join(dir_path, filename) if dir_path.endswith(os.sep) else dir_path + filename

def _read_parquet_glob(spark, folder_path: str, label: str):
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    if not files_list:
        raise FileNotFoundError(f"[{label}] No parquet files found in: {folder_path}")
    df = spark.read.option("header", "true").parquet(*files_list)
    print(f"[LOAD] {label}: {len(files_list)} file(s) from {folder_path} rows={df.count()}")
    return df

def _expect_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")

# ------------------------------
# A) PRETRAIN FEATURE STORE (consumes separate Gold sources)
# ------------------------------
def build_gold_pretrain_feature_store_from_gold(
    snapshot_date_str: str,
    gold_label_store_directory: str,
    gold_feature_attributes_directory: str,
    gold_feature_clickstream_directory: str,
    gold_feature_financials_directory: str,
    gold_pretrain_feature_store_directory: str,
    spark,
    mob: int,
    dedupe_strategy: str = "latest",     # "latest" | "none"
    drop_null_feature_rows: bool = False
):
    """
    Build pretrain features for one snapshot date by consuming three separate Gold feature stores:
      - Attributes, Clickstream, Financials
    Assumptions:
      * All sources (and labels) have a column named 'snapshot_date'.
      * Labels live at: gold_label_store_<YYYY_MM_DD>.parquet
      * Feature sources are directories of parquet (already 'Gold', not Silver).

    Steps:
      1) Load label store for the given snapshot date and rename snapshot_date -> label_snapshot_date.
      2) Join each source to labels on Customer_ID.
      3) Filter each source to rows with source.snapshot_date < add_months(label_snapshot_date, -mob + 1).
      4) For clickstream, aggregate mean(fe_1..fe_20) -> fe_i_mean.
      5) Merge all into one features DF and save as gold_pretrain_feature_store_<date>.parquet
    """
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # 1) Labels
    label_file = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    labels_path = _cat_path(gold_label_store_directory, label_file)
    labels_df = spark.read.parquet(labels_path).withColumnRenamed("snapshot_date", "label_snapshot_date")
    _expect_cols(labels_df, ["Customer_ID", "label_snapshot_date"], "labels_df")
    print(f"[LOAD] labels: {labels_path} rows={labels_df.count()}")

    # 2) Attributes
    attr_df = (
        _read_parquet_glob(spark, gold_feature_attributes_directory, label="attributes")
        .alias("attr")
        .join(labels_df.alias("lbl"), on="Customer_ID", how="inner")
        .filter(col("attr.snapshot_date") < F.add_months(col("lbl.label_snapshot_date"), -mob + 1))
        .select(
            col("attr.Customer_ID"),
            col("attr.Age_bin"),
            col("attr.Occupation"),
            col("attr.snapshot_date").alias("attributes_snapshot_date"),
            col("lbl.label_snapshot_date"),
        )
    )
    print(f"[ATTR] after cutoff rows={attr_df.count()}")

    # 3) Financials
    fin_raw = (
        _read_parquet_glob(spark, gold_feature_financials_directory, label="financials")
        .alias("fin")
        .join(labels_df.alias("lbl"), on="Customer_ID", how="inner")
        .filter(col("fin.snapshot_date") < F.add_months(col("lbl.label_snapshot_date"), -mob + 1))
    )

    fin_keep = [
        "Customer_ID",
        "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Type_of_Loan",
        "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Credit_Mix", "Payment_of_Min_Amount",
        "Payment_Behaviour_Spent", "Payment_Behaviour_Payment",
    ]
    # keep engineered/encodings if present
    for extra in [
        "emi_to_income_ratio", "debt_to_income_ratio", "avg_delay", "balance_to_income_ratio",
        "high_credit_inquiry_flag", "high_utilization_flag", "high_emi_burden_flag", "negative_balance_flag",
        "Credit_Mix_Enc", "Payment_of_Min_Amount_Enc", "Payment_Behaviour_Spent_Enc", "Payment_Behaviour_Payment_Enc"
    ]:
        if extra in fin_raw.columns:
            fin_keep.append(extra)

    fin_df = fin_raw.select(
        *[col(f"fin.{c}") for c in fin_keep],
        col("fin.snapshot_date").alias("financials_snapshot_date"),
        col("lbl.label_snapshot_date"),
    )
    print(f"[FIN]  after cutoff rows={fin_df.count()}")

    # 4) Clickstream (mean agg)
    cs_raw = (
        _read_parquet_glob(spark, gold_feature_clickstream_directory, label="cickstream")
        .alias("cs")
        .join(labels_df.alias("lbl"), on="Customer_ID", how="inner")
        .filter(col("cs.snapshot_date") < F.add_months(col("lbl.label_snapshot_date"), -mob + 1))
    )
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    _expect_cols(cs_raw, fe_cols, "clickstream_df")

    cs_df = (
        cs_raw
        .select("cs.Customer_ID", *[col(f"cs.{c}") for c in fe_cols])
        .groupby("Customer_ID")
        .agg(*[F.round(F.mean(c), 2).alias(f"{c}_mean") for c in fe_cols])
    )
    print(f"[CLKS] post-mean rows={cs_df.count()}")

    # 5) Merge
    feature_df = cs_df.join(attr_df, on="Customer_ID", how="inner") \
                      .join(fin_df,  on=["Customer_ID", "label_snapshot_date"], how="inner")

    # If encodings weren’t already in financials gold, add them now:
    if "Credit_Mix_Enc" not in feature_df.columns and "Credit_Mix" in feature_df.columns:
        feature_df = feature_df.withColumn(
            "Credit_Mix_Enc",
            F.when(col("Credit_Mix") == "Bad", 0)
             .when(col("Credit_Mix") == "Standard", 1)
             .when(col("Credit_Mix") == "Good", 2)
             .otherwise(None).cast(IntegerType())
        )
    if "Payment_Behaviour_Spent_Enc" not in feature_df.columns and "Payment_Behaviour_Spent" in feature_df.columns:
        feature_df = feature_df.withColumn(
            "Payment_Behaviour_Spent_Enc",
            F.when(col("Payment_Behaviour_Spent") == "Low", 0)
             .when(col("Payment_Behaviour_Spent") == "High", 1)
             .otherwise(None).cast(IntegerType())
        )
    if "Payment_Behaviour_Payment_Enc" not in feature_df.columns and "Payment_Behaviour_Payment" in feature_df.columns:
        feature_df = feature_df.withColumn(
            "Payment_Behaviour_Payment_Enc",
            F.when(col("Payment_Behaviour_Payment") == "Small", 0)
             .when(col("Payment_Behaviour_Payment") == "Medium", 1)
             .when(col("Payment_Behaviour_Payment") == "Large", 2)
             .otherwise(None).cast(IntegerType())
        )
    if "Payment_of_Min_Amount_Enc" not in feature_df.columns and "Payment_of_Min_Amount" in feature_df.columns:
        feature_df = feature_df.withColumn(
            "Payment_of_Min_Amount_Enc",
            F.when(col("Payment_of_Min_Amount") == "Yes", 1)
             .when(col("Payment_of_Min_Amount") == "No", 0)
             .otherwise(None).cast(IntegerType())
        )

    # Optional dedupe (one row per customer, prefer freshest financials/attributes snapshots)
    if dedupe_strategy == "latest":
        date_cols = [c for c in ["financials_snapshot_date", "attributes_snapshot_date"] if c in feature_df.columns]
        if date_cols:
            w = Window.partitionBy("Customer_ID").orderBy(*[col(c).desc_nulls_last() for c in date_cols])
            feature_df = feature_df.withColumn("_rn", F.row_number().over(w)).filter(col("_rn") == 1).drop("_rn")
            print("[DEDUPE] kept latest snapshot per Customer_ID")
        else:
            print("[DEDUPE] skipped (no *_snapshot_date columns found)")
    elif dedupe_strategy != "none":
        raise ValueError("dedupe_strategy must be 'latest' or 'none'")

    # Optional null-drop for model features (keep metadata)
    if drop_null_feature_rows:
        meta_cols = {
            "Customer_ID", "label_snapshot_date",
            "attributes_snapshot_date", "financials_snapshot_date",
            "Age_bin", "Occupation"
        }
        feature_cols = [c for c in feature_df.columns if c not in meta_cols]
        cond = None
        for c in feature_cols:
            cnd = col(c).isNotNull()
            cond = cnd if cond is None else (cond & cnd)
        feature_df = feature_df.filter(cond)
        print("[CLEAN] dropped rows with NULLs in feature columns")

    # Column order
    ordered_cols = [
        "Customer_ID", "label_snapshot_date",
        "Age_bin", "Occupation", "attributes_snapshot_date",
        "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Type_of_Loan",
        "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "financials_snapshot_date",
        # Encodings / engineered / flags if present
        *[c for c in ["Credit_Mix_Enc", "Payment_of_Min_Amount_Enc",
                      "Payment_Behaviour_Spent_Enc", "Payment_Behaviour_Payment_Enc",
                      "emi_to_income_ratio", "debt_to_income_ratio", "avg_delay", "balance_to_income_ratio",
                      "high_credit_inquiry_flag", "high_utilization_flag",
                      "high_emi_burden_flag", "negative_balance_flag"] if c in feature_df.columns],
        # Clickstream means
        *[f"fe_{i}_mean" for i in range(1, 21)],
    ]
    ordered_cols = [c for c in ordered_cols if c in feature_df.columns]
    feature_df = feature_df.select(*ordered_cols)

    # Save
    out_file = f"gold_pretrain_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = _cat_path(gold_pretrain_feature_store_directory, out_file)
    feature_df.write.mode("overwrite").parquet(out_path)
    print(f"[SAVE] pretrain features -> {out_path} rows={feature_df.count()}")

    return feature_df

# ------------------------------
# B) TRAINING TABLE (features + labels)
# ------------------------------
def build_gold_training_table_from_pretrain(
    snapshot_date_str: str,
    gold_pretrain_feature_store_directory: str,
    gold_label_store_directory: str,
    gold_training_store_directory: str,
    spark,
    drop_null_feature_rows: bool = False
):
    """
    Join the pretrain feature store with the label store for the same snapshot date.
    Join keys: Customer_ID AND label_snapshot_date == snapshot_date
    Output: gold_training_store_<date>.parquet
    """
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    feat_file = f"gold_pretrain_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    lab_file  = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"

    feat_path = _cat_path(gold_pretrain_feature_store_directory, feat_file)
    lab_path  = _cat_path(gold_label_store_directory, lab_file)

    features_df = spark.read.parquet(feat_path)
    labels_df   = spark.read.parquet(lab_path)

    _expect_cols(features_df, ["Customer_ID", "label_snapshot_date"], "features_df")
    _expect_cols(labels_df,   ["Customer_ID", "snapshot_date", "label", "label_def", "loan_id"], "labels_df")

    joined_df = (
        features_df.alias("feat")
        .join(
            labels_df.alias("lab"),
            on=[features_df.Customer_ID == labels_df.Customer_ID,
                features_df.label_snapshot_date == labels_df.snapshot_date],
            how="inner"
        )
        .drop(labels_df.Customer_ID)
        .drop(labels_df.snapshot_date)
    )

    if drop_null_feature_rows:
        exclude = {"loan_id", "label", "label_def", "label_snapshot_date", "Customer_ID"}
        feature_cols = [c for c in joined_df.columns if c not in exclude]
        cond = None
        for c in feature_cols:
            cnd = F.col(c).isNotNull()
            cond = cnd if cond is None else (cond & cnd)
        joined_df = joined_df.filter(cond)

    out_file = f"gold_training_store_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = _cat_path(gold_training_store_directory, out_file)
    joined_df.write.mode("overwrite").parquet(out_path)
    print(f"[SAVE] training table -> {out_path} rows={joined_df.count()}")
    return joined_df
