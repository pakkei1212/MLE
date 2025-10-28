# utils/data_processing_gold_table.py

import os, glob
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType


def _silver_part(snapshot_date_str: str, prefix: str) -> str:
    # e.g. prefix='silver_clickstream_' -> 'silver_clickstream_2023_03_01.parquet'
    return f"{prefix}{snapshot_date_str.replace('-', '_')}.parquet"

def _read_all_parquet(spark, folder_path: str):
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    if not files_list:
        raise FileNotFoundError(f"No parquet files found in: {folder_path}")
    df = spark.read.option("header", "true").parquet(*files_list)
    print('loaded from:', files_list, 'row count:', df.count())
    return df

# ------------------------------
# A) LABEL STORE (Gold)
# ------------------------------
def build_gold_label_store(snapshot_date_str,
                           silver_loan_daily_directory,
                           gold_label_store_directory,
                           spark,
                           dpd: int,
                           mob: int):
    """
    Reads Silver loan_daily, filters target MOB, creates labels (dpd>=threshold),
    writes gold_label_store_YYYY_MM_DD.parquet and returns the DataFrame.
    """
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    part = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + ".parquet"
    loans_path = silver_loan_daily_directory + part
    loans_df = spark.read.parquet(loans_path)
    print('loaded from:', loans_path, 'row count:', loans_df.count())

    # target MOB
    loans_df = loans_df.filter(col("mob") == mob)

    # label
    loans_df = loans_df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    loans_df = loans_df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    # keep minimal columns
    labels_df = loans_df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    out_labels = "gold_label_store_" + snapshot_date_str.replace('-', '_') + ".parquet"
    labels_path = gold_label_store_directory + out_labels
    labels_df.write.mode("overwrite").parquet(labels_path)
    print('saved to:', labels_path, "| Labels:", labels_df.count())

    return labels_df

# ------------------------------
# B) FEATURE STORE (Gold)
# ------------------------------
def build_gold_feature_clickstream(snapshot_date_str: str,
                                   silver_clickstream_directory: str,
                                   gold_feature_store_directory: str,
                                   spark):
    """
    Read the single Silver clickstream parquet for the snapshot date (no aggregation),
    and write a Gold clickstream feature table for the same date.
    Input file name expected:
        {silver_clickstream_directory}/silver_clickstream_YYYY_MM_DD.parquet
    Output file name:
        {gold_feature_store_directory}/gold_feature_clickstream_YYYY_MM_DD.parquet
    """
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    part = "silver_users_clickstream_" + snapshot_date_str.replace('-', '_') + ".parquet"
    src_path = os.path.join(silver_clickstream_directory, part)

    cs_df = spark.read.parquet(src_path)
    print('loaded from:', src_path, 'row count:', cs_df.count())

    # Ensure we keep Customer_ID and all fe_* columns as-is; add a feature_snapshot_date marker.
    fe_cols = [c for c in cs_df.columns if c.startswith("fe_")]
    keep_cols = ["Customer_ID", "snapshot_date"] + fe_cols
    cs_out = (
        cs_df.select(*keep_cols)
             .withColumn("feature_snapshot_date", F.to_date(F.lit(snapshot_date_str), "yyyy-MM-dd"))
    )

    out_name = f"gold_feature_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_name)
    cs_out.write.mode("overwrite").parquet(out_path)
    print(f"[Gold-Clickstream] rows={cs_out.count()} saved={out_path}")
    return cs_out


def build_gold_feature_attributes(snapshot_date_str: str,
                                  silver_attributes_directory: str,
                                  gold_feature_store_directory: str,
                                  spark):
    """
    Read the single Silver attributes parquet for the snapshot date (no aggregation),
    bin Age, keep Occupation, and write Gold attributes.
    Input:
        {silver_attributes_directory}/silver_attributes_YYYY_MM_DD.parquet
    Output:
        {gold_feature_store_directory}/gold_feature_attributes_YYYY_MM_DD.parquet
    """
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    part = _silver_part(snapshot_date_str, "silver_users_attributes_")
    src_path = os.path.join(silver_attributes_directory, part)

    attr_df = spark.read.parquet(src_path).select("Customer_ID", "Age", "Occupation", "snapshot_date")
    attr_out = (
        attr_df
        .withColumn(
            "Age_bin",
            F.when(col("Age") < 18, "under_18")
             .when((col("Age") >= 18) & (col("Age") < 25), "18_24")
             .when((col("Age") >= 25) & (col("Age") < 35), "25_34")
             .when((col("Age") >= 35) & (col("Age") < 50), "35_49")
             .when((col("Age") >= 50) & (col("Age") < 65), "50_64")
             .when((col("Age") >= 65), "65_plus")
             .otherwise(None)
        )
    )

    out_name = f"gold_feature_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_name)
    attr_out.write.mode("overwrite").parquet(out_path)
    print(f"[Gold-Attributes] rows={attr_out.count()} saved={out_path}")
    return attr_out


def build_gold_feature_financials(snapshot_date_str: str,
                                  silver_financials_directory: str,
                                  gold_feature_store_directory: str,
                                  spark):
    """
    Read the single Silver financials parquet for the snapshot date (no aggregation),
    apply encodings, ratios, and flags; then write Gold financials.
    Input:
        {silver_financials_directory}/silver_financials_YYYY_MM_DD.parquet
    Output:
        {gold_feature_store_directory}/gold_feature_financials_YYYY_MM_DD.parquet
    """
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    part = _silver_part(snapshot_date_str, "silver_users_financials_")
    src_path = os.path.join(silver_financials_directory, part)

    fin_df = spark.read.parquet(src_path).select(
        "Customer_ID",
        "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Credit_Mix", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age",
        "Payment_of_Min_Amount", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance",
        "Payment_Behaviour_Spent", "Payment_Behaviour_Payment",
        "snapshot_date"
    )

    fin_feat = (
        fin_df        
        # Encodings
        .withColumn(
            "Credit_Mix_Enc",
            F.when(col("Credit_Mix") == "Bad", 0)
             .when(col("Credit_Mix") == "Standard", 1)
             .when(col("Credit_Mix") == "Good", 2)
             .otherwise(None).cast(IntegerType())
        )
        .withColumn(
            "Payment_Behaviour_Spent_Enc",
            F.when(col("Payment_Behaviour_Spent") == "Low", 0)
             .when(col("Payment_Behaviour_Spent") == "High", 1)
             .otherwise(None).cast(IntegerType())
        )
        .withColumn(
            "Payment_Behaviour_Payment_Enc",
            F.when(col("Payment_Behaviour_Payment") == "Small", 0)
             .when(col("Payment_Behaviour_Payment") == "Medium", 1)
             .when(col("Payment_Behaviour_Payment") == "Large", 2)
             .otherwise(None).cast(IntegerType())
        )
        .withColumn(
            "Payment_of_Min_Amount_Enc",
            F.when(col("Payment_of_Min_Amount") == "Yes", 1)
             .when(col("Payment_of_Min_Amount") == "No", 0)
             .otherwise(None).cast(IntegerType())
        )
        # Ratios
        .withColumn(
            "avg_delay",
            F.when(col("Num_of_Loan") > 0, col("Delay_from_due_date") / col("Num_of_Loan")).otherwise(None)
        )
        .withColumn(
            "balance_to_income_ratio",
            F.when(col("Annual_Income") > 0, col("Monthly_Balance") / (col("Annual_Income") / 12)).otherwise(None)
        )
        .withColumn(
            "emi_to_income_ratio",
            F.when(col("Annual_Income") > 0, col("Total_EMI_per_month") / (col("Annual_Income") / 12)).otherwise(None)
        )
        .withColumn(
            "debt_to_income_ratio",
            F.when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income")).otherwise(None)
        )
        # Flags
        .withColumn("high_credit_inquiry_flag", F.when(col("Num_Credit_Inquiries") > 10, 1).otherwise(0))
        .withColumn("high_utilization_flag",     F.when(col("Credit_Utilization_Ratio") > 0.8, 1).otherwise(0))
        .withColumn("high_emi_burden_flag",      F.when(col("emi_to_income_ratio") > 0.5, 1).otherwise(0))
        .withColumn("negative_balance_flag",     F.when(col("Monthly_Balance") < 0, 1).otherwise(0))
    )

    fin_cols = [
        "Customer_ID", "snapshot_date",
        "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Type_of_Loan",
        "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour_Spent", "Payment_Behaviour_Payment",
        "Credit_Mix_Enc", "Payment_of_Min_Amount_Enc",
        "Payment_Behaviour_Spent_Enc", "Payment_Behaviour_Payment_Enc",
        "emi_to_income_ratio", "debt_to_income_ratio",
        "avg_delay", "balance_to_income_ratio",
        "high_credit_inquiry_flag", "high_utilization_flag",
        "high_emi_burden_flag", "negative_balance_flag",
    ]
    fin_out = fin_feat.select(*fin_cols)

    out_name = f"gold_feature_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_name)
    fin_out.write.mode("overwrite").parquet(out_path)
    print(f"[Gold-Financials] rows={fin_out.count()} saved={out_path}")
    return fin_out

'''def build_gold_feature_store(snapshot_date_str,
                             gold_label_store_directory,
                             silver_clickstream_directory,
                             silver_attributes_directory,
                             silver_financials_directory,
                             gold_feature_store_directory,
                             spark,
                             mob: int):
    """
    Reads Gold label store for the same snapshot date to get (Customer_ID, label_snapshot_date),
    joins with Silver user tables (pre-label window), aggregates/engineers, and writes feature store.
    """
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Read label store to drive cutoff & user list
    labels_part = "gold_label_store_" + snapshot_date_str.replace('-', '_') + ".parquet"
    labels_path = gold_label_store_directory + labels_part
    labels_df = spark.read.parquet(labels_path).withColumnRenamed("snapshot_date", "label_snapshot_date")
    print("loaded labels from:", labels_path, "rows:", labels_df.count())

    # ---------- Clickstream (means fe_1..fe_20 before label window) ----------
    cs_df = _read_all_parquet(spark, silver_clickstream_directory)
    cs_df = (
        cs_df.alias("scd")
        .join(labels_df.alias("lbl"), on="Customer_ID", how="inner")
        .filter(F.col("scd.snapshot_date") < F.add_months(F.col("lbl.label_snapshot_date"), -mob + 1))
        .select("scd.Customer_ID", *[F.col(f"scd.fe_{i}") for i in range(1, 21)])
        .groupby("Customer_ID")
        .agg(*[F.round(F.mean(f"fe_{i}"), 2).alias(f"fe_{i}_mean") for i in range(1, 21)])
    )
    print("Clickstream (post-agg):", cs_df.count())

    # ---------- Attributes (bin Age, keep Occupation) ----------
    attr_df = _read_all_parquet(spark, silver_attributes_directory)
    attr_df = (
        attr_df.alias("sad")
        .join(labels_df.alias("lbl"), on="Customer_ID", how="inner")
        .filter(F.col("sad.snapshot_date") < F.add_months(F.col("lbl.label_snapshot_date"), -mob + 1))
        .select(
            F.col("sad.Customer_ID"),
            F.col("sad.Age"),
            F.col("sad.Occupation"),
            F.col("sad.snapshot_date").alias("attributes_snapshot_date"),
            F.col("lbl.label_snapshot_date"),
        )
    )
    attr_df = attr_df.withColumn(
        "Age_bin",
        F.when(F.col("Age") < 18, "under_18")
         .when((F.col("Age") >= 18) & (F.col("Age") < 25), "18_24")
         .when((F.col("Age") >= 25) & (F.col("Age") < 35), "25_34")
         .when((F.col("Age") >= 35) & (F.col("Age") < 50), "35_49")
         .when((F.col("Age") >= 50) & (F.col("Age") < 65), "50_64")
         .when((F.col("Age") >= 65), "65_plus")
         .otherwise(None)
    ).select("Customer_ID", "Age_bin", "Occupation", "attributes_snapshot_date", "label_snapshot_date")
    print("Attributes (post-bin):", attr_df.count())

    # ---------- Financials (raw + splits) ----------
    fin_df = _read_all_parquet(spark, silver_financials_directory)
    fin_df = (
        fin_df.alias("sfd")
        .join(labels_df.alias("lbl"), on="Customer_ID", how="inner")
        .filter(F.col("sfd.snapshot_date") < F.add_months(F.col("lbl.label_snapshot_date"), -mob + 1))
        .select(
            F.col("sfd.Customer_ID"),
            F.col("sfd.Annual_Income"),
            F.col("sfd.Monthly_Inhand_Salary"),
            F.col("sfd.Num_Bank_Accounts"),
            F.col("sfd.Num_Credit_Card"),
            F.col("sfd.Interest_Rate"),
            F.col("sfd.Num_of_Loan"),
            F.col("sfd.Type_of_Loan"),
            F.col("sfd.Delay_from_due_date"),
            F.col("sfd.Num_of_Delayed_Payment"),
            F.col("sfd.Changed_Credit_Limit"),
            F.col("sfd.Num_Credit_Inquiries"),
            F.col("sfd.Credit_Mix"),
            F.col("sfd.Outstanding_Debt"),
            F.col("sfd.Credit_Utilization_Ratio"),
            F.col("sfd.Credit_History_Age"),
            F.col("sfd.Payment_of_Min_Amount"),
            F.col("sfd.Total_EMI_per_month"),
            F.col("sfd.Amount_invested_monthly"),
            F.col("sfd.Monthly_Balance"),
            F.col("sfd.Payment_Behaviour_Spent"),
            F.col("sfd.Payment_Behaviour_Payment"),
            F.col("sfd.snapshot_date").alias("financials_snapshot_date"),
            F.col("lbl.label_snapshot_date"),
        )
    )
    print("Financials (post-filter):", fin_df.count())

    # ---------- Merge + encodings + engineered features ----------
    feature_df = cs_df.join(attr_df, on="Customer_ID", how="inner") \
                      .join(fin_df, on="Customer_ID", how="inner")

    # Encodings
    feature_df = feature_df.withColumn(
        "Credit_Mix_Enc",
        F.when(col("Credit_Mix") == "Bad", 0)
         .when(col("Credit_Mix") == "Standard", 1)
         .when(col("Credit_Mix") == "Good", 2)
         .otherwise(None).cast(IntegerType())
    ).withColumn(
        "Payment_Behaviour_Spent_Enc",
        F.when(col("Payment_Behaviour_Spent") == "Low", 0)
         .when(col("Payment_Behaviour_Spent") == "High", 1)
         .otherwise(None).cast(IntegerType())
    ).withColumn(
        "Payment_Behaviour_Payment_Enc",
        F.when(col("Payment_Behaviour_Payment") == "Small", 0)
         .when(col("Payment_Behaviour_Payment") == "Medium", 1)
         .when(col("Payment_Behaviour_Payment") == "Large", 2)
         .otherwise(None).cast(IntegerType())
    ).withColumn(
        "Payment_of_Min_Amount_Enc",
        F.when(col("Payment_of_Min_Amount") == "Yes", 1)
         .when(col("Payment_of_Min_Amount") == "No", 0)
         .otherwise(None).cast(IntegerType())
    )

    # Ratios
    feature_df = feature_df.withColumn(
        "avg_delay",
        F.when(col("Num_of_Loan") > 0, col("Delay_from_due_date") / col("Num_of_Loan"))
         .otherwise(None)
    ).withColumn(
        "balance_to_income_ratio",
        F.when(col("Annual_Income") > 0, col("Monthly_Balance") / (col("Annual_Income") / 12))
         .otherwise(None)
    ).withColumn(
        "emi_to_income_ratio",
        F.when(col("Annual_Income") > 0, col("Total_EMI_per_month") / (col("Annual_Income") / 12))
         .otherwise(None)
    ).withColumn(
        "debt_to_income_ratio",
        F.col("Outstanding_Debt") / F.col("Annual_Income")
    )

    # Flags
    feature_df = feature_df.withColumn("high_credit_inquiry_flag", F.when(col("Num_Credit_Inquiries") > 10, 1).otherwise(0)) \
                           .withColumn("high_utilization_flag",     F.when(col("Credit_Utilization_Ratio") > 0.8, 1).otherwise(0)) \
                           .withColumn("high_emi_burden_flag",      F.when(col("emi_to_income_ratio") > 0.5, 1).otherwise(0)) \
                           .withColumn("negative_balance_flag",     F.when(col("Monthly_Balance") < 0, 1).otherwise(0))

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
        "Credit_Mix_Enc", "Payment_of_Min_Amount_Enc",
        "Payment_Behaviour_Spent_Enc", "Payment_Behaviour_Payment_Enc",
        "emi_to_income_ratio", "debt_to_income_ratio",
        "avg_delay", "balance_to_income_ratio",
        "high_credit_inquiry_flag", "high_utilization_flag",
        "high_emi_burden_flag", "negative_balance_flag",
        *[f"fe_{i}_mean" for i in range(1, 21)],
    ]
    feature_df = feature_df.select(*ordered_cols)

    out_features = "gold_feature_store_" + snapshot_date_str.replace('-', '_') + ".parquet"
    features_path = gold_feature_store_directory + out_features
    feature_df.write.mode("overwrite").parquet(features_path)
    print('saved to:', features_path, "| Features:", feature_df.count())
    return feature_df


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, silver_clickstream_directory, 
                              silver_attributes_directory, silver_financials_directory, 
                              gold_feature_store_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    #============================================
    # labels
    #============================================     
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    loans_df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', loans_df.count())

    # get customer at mob
    loans_df = loans_df.filter(col("mob") == mob)

    # get label
    loans_df = loans_df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    loans_df = loans_df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    loans_df = loans_df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    loans_df.write.mode("overwrite").parquet(filepath)
    # loans_df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    print("Loans:", loans_df.count())

    #============================================
    # features - user clickstream
    #============================================   
    # connect to silver table
    folder_path = silver_clickstream_directory
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    clickstream_df = spark.read.option("header", "true").parquet(*files_list)
    print('loaded from:', files_list, 'row count:', clickstream_df.count())

    feature_cols = [F.col(f"scd.fe_{i}") for i in range(1, 21)]

    # join with loans to filter only users with loans and before label snapshot date
    clickstream_df = (
        clickstream_df.alias("scd")
        .join(loans_df.alias("gcd"), on="Customer_ID", how="inner")
        .filter(F.col("scd.snapshot_date") < F.add_months(F.col("gcd.snapshot_date"), -mob + 1))
        .select(
            "scd.Customer_ID",
            *feature_cols
        )
    )

    # aggregate features by user - mean
    clickstream_df = clickstream_df.groupby("Customer_ID").agg(
        *[F.round(F.mean(f"fe_{i}"), 2).alias(f"fe_{i}_mean") for i in range(1, 21)]
    )

    #============================================
    # features - user attributes
    #============================================   
    # connect to silver table
    folder_path = silver_attributes_directory
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    attributes_df = spark.read.option("header", "true").parquet(*files_list)
    print('loaded from:', files_list, 'row count:', attributes_df.count())

    # join with loans to filter only users with loans and before label snapshot date
    attributes_df = (
        attributes_df.alias("sad")
        .join(loans_df.alias("gcd"), on="Customer_ID", how="inner")
        .filter(F.col("sad.snapshot_date") < F.add_months(F.col("gcd.snapshot_date"), -mob + 1))
        .select(
            F.col("sad.Customer_ID"),
            F.col("sad.Age"),
            F.col("sad.Occupation"),
            F.col("sad.snapshot_date").alias("attributes_snapshot_date")
        )
    )

    # bin age
    attributes_df = attributes_df.withColumn(
        "Age_bin",
        F.when(F.col("Age") < 18, "under_18")
        .when((F.col("Age") >= 18) & (F.col("Age") < 25), "18_24")
        .when((F.col("Age") >= 25) & (F.col("Age") < 35), "25_34")
        .when((F.col("Age") >= 35) & (F.col("Age") < 50), "35_49")
        .when((F.col("Age") >= 50) & (F.col("Age") < 65), "50_64")
        .when((F.col("Age") >= 65), "65_plus")
        .otherwise(None)
    )

    # select columns to save
    attributes_df = attributes_df.select("Customer_ID", "Age_bin", "Occupation", "attributes_snapshot_date")

    #============================================
    # features - user financials
    #============================================   
    # connect to silver table
    folder_path = silver_financials_directory
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    financials_df = spark.read.option("header", "true").parquet(*files_list)
    print('loaded from:', files_list, 'row count:', financials_df.count())

    financials_df = (
        financials_df.alias("sfd")
        .join(loans_df.alias("gcd"), on="Customer_ID", how="inner")
        .filter(F.col("sfd.snapshot_date") < F.add_months(F.col("gcd.snapshot_date"), -mob + 1))
        .select(
            F.col("sfd.Customer_ID"),
            F.col("sfd.Annual_Income"),
            F.col("sfd.Monthly_Inhand_Salary"),
            F.col("sfd.Num_Bank_Accounts"),
            F.col("sfd.Num_Credit_Card"),
            F.col("sfd.Interest_Rate"),
            F.col("sfd.Num_of_Loan"),
            F.col("sfd.Type_of_Loan"),
            F.col("sfd.Delay_from_due_date"),
            F.col("sfd.Num_of_Delayed_Payment"),
            F.col("sfd.Changed_Credit_Limit"),
            F.col("sfd.Num_Credit_Inquiries"),
            F.col("sfd.Credit_Mix"),
            F.col("sfd.Outstanding_Debt"),
            F.col("sfd.Credit_Utilization_Ratio"),
            F.col("sfd.Credit_History_Age"),
            F.col("sfd.Payment_of_Min_Amount"),
            F.col("sfd.Total_EMI_per_month"),
            F.col("sfd.Amount_invested_monthly"),
            F.col("sfd.Monthly_Balance"),
            F.col("sfd.Payment_Behaviour_Spent"),
            F.col("sfd.Payment_Behaviour_Payment"),
            F.col("sfd.snapshot_date").alias("financials_snapshot_date"),
            F.col("gcd.snapshot_date").alias("label_snapshot_date")
        )
    )

    feature_df = clickstream_df.join(attributes_df, on="Customer_ID", how="inner") \
                               .join(financials_df, on="Customer_ID", how="inner")
    
    # encode categorical features
    # Encode Credit_Mix
    feature_df = feature_df.withColumn(
        "Credit_Mix_Enc",
        F.when(col("Credit_Mix") == "Bad", 0)
        .when(col("Credit_Mix") == "Standard", 1)
        .when(col("Credit_Mix") == "Good", 2)
        .otherwise(None).cast(IntegerType())
    )

    # Encode Payment_Behaviour_Spent
    feature_df = feature_df.withColumn(
        "Payment_Behaviour_Spent_Enc",
        F.when(col("Payment_Behaviour_Spent") == "Low", 0)
        .when(col("Payment_Behaviour_Spent") == "High", 1)
        .otherwise(None).cast(IntegerType())
    )

    # Encode Payment_Behaviour_Payment
    feature_df = feature_df.withColumn(
        "Payment_Behaviour_Payment_Enc",
        F.when(col("Payment_Behaviour_Payment") == "Small", 0)
        .when(col("Payment_Behaviour_Payment") == "Medium", 1)
        .when(col("Payment_Behaviour_Payment") == "Large", 2)
        .otherwise(None).cast(IntegerType())
    )

    # Encode Payment_of_Min_Amount
    feature_df = feature_df.withColumn(
        "Payment_of_Min_Amount_Enc",
        F.when(col("Payment_of_Min_Amount") == "Yes", 1)
        .when(col("Payment_of_Min_Amount") == "No", 0)
        .otherwise(None).cast(IntegerType())
    )
    
    # add engineered features by business logic
    # --- Derived ratios ---
    feature_df = (
        feature_df
        # avg_delay = Delay_from_due_date / Num_of_Loan
        .withColumn(
            "avg_delay",
            F.when(col("Num_of_Loan") > 0, col("Delay_from_due_date") / col("Num_of_Loan"))
            .otherwise(None)
        )
        # balance_to_income_ratio = Monthly_Balance / (Annual_Income / 12)
        .withColumn(
            "balance_to_income_ratio",
            F.when(col("Annual_Income") > 0, col("Monthly_Balance") / (col("Annual_Income") / 12))
            .otherwise(None)
        )
        # emi_to_income_ratio = Total_EMI_per_month / (Annual_Income / 12)
        .withColumn(
            "emi_to_income_ratio",
            F.when(col("Annual_Income") > 0, col("Total_EMI_per_month") / (col("Annual_Income") / 12))
            .otherwise(None)
        )
        # debt_to_income_ratio = Outstanding_Debt / Annual_Income
        .withColumn(
            "debt_to_income_ratio",
            F.col("Outstanding_Debt") / F.col("Annual_Income")
        )
    )

    # --- Risk flags ---
    feature_df = (
        feature_df
        # high_credit_inquiry_flag = (Num_Credit_Inquiries > 10)
        .withColumn(
            "high_credit_inquiry_flag",
            F.when(col("Num_Credit_Inquiries") > 10, 1).otherwise(0)
        )
        # high_utilization_flag = (Credit_Utilization_Ratio > 0.8)
        .withColumn(
            "high_utilization_flag",
            F.when(col("Credit_Utilization_Ratio") > 0.8, 1).otherwise(0)
        )
        # high_emi_burden_flag = (emi_to_income_ratio > 0.5)
        .withColumn(
            "high_emi_burden_flag",
            F.when(col("emi_to_income_ratio") > 0.5, 1).otherwise(0)
        )
        # negative_balance_flag = (Monthly_Balance < 0)
        .withColumn(
            "negative_balance_flag",
            F.when(col("Monthly_Balance") < 0, 1).otherwise(0)
        )
    )

    # Define the order explicitly
    ordered_cols = [
        # Keys / metadata
        "Customer_ID", "label_snapshot_date",
        
        # Attributes
        "Age_bin", "Occupation", "attributes_snapshot_date",
        
        # Financials (raw numeric + engineered ratios)
        "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Type_of_Loan",
        "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "financials_snapshot_date",
        
        # Encoded categorical features
        "Credit_Mix_Enc", 
        "Payment_of_Min_Amount_Enc",
        "Payment_Behaviour_Spent_Enc", 
        "Payment_Behaviour_Payment_Enc",
        
        # Engineered ratios
        "emi_to_income_ratio", "debt_to_income_ratio", 
        "avg_delay", "balance_to_income_ratio",
        
        # Flags
        "high_credit_inquiry_flag", "high_utilization_flag",
        "high_emi_burden_flag", "negative_balance_flag"
    ]

    # Add clickstream features (fe_1_mean to fe_20_mean)
    clickstream_cols = [f"fe_{i}_mean" for i in range(1, 21)]

    # Combine all
    ordered_cols.extend(clickstream_cols)

    # Reorder feature_df
    feature_df = feature_df.select(*ordered_cols)

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    feature_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    print("Clickstream:", clickstream_df.count())
    print("Attributes:", attributes_df.count())
    print("Financials:", financials_df.count())
    print("Features:", feature_df.count())
    
    return loans_df, feature_df'''