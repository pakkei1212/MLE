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
    
    return loans_df, feature_df