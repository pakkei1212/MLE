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


def process_bronze_loans(snapshot_date_str, bronze_lms_directory, spark, csv_file_path="data/lms_loan_daily.csv"):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

# 1) clickstream
def process_bronze_users_clickstream(
    snapshot_date_str,
    bronze_clickstream_directory,
    spark,
    clickstream_csv="data/feature_clickstream.csv",
):
    # prepare arguments (kept for parity with original)
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # clickstream
    clickstream_df = (
        spark.read.csv(clickstream_csv, header=True, inferSchema=True)
        .filter(col("snapshot_date") == snapshot_date_str)
    )
    print(snapshot_date_str + " row count (clickstream):", clickstream_df.count())

    partition_name = "bronze_users_clickstream_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_clickstream_directory + partition_name
    clickstream_df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return clickstream_df

# 2) attributes
def process_bronze_users_attributes(
    snapshot_date_str,
    bronze_attributes_directory,
    spark,
    attributes_csv="data/features_attributes.csv",
):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    attributes_df = spark.read.csv(attributes_csv, header=True, inferSchema=True)
    if "snapshot_date" in attributes_df.columns:
        attributes_df = attributes_df.filter(col("snapshot_date") == snapshot_date_str)

    print(snapshot_date_str + " row count (attributes):", attributes_df.count())

    partition_name = "bronze_users_attributes_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_attributes_directory + partition_name
    attributes_df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return attributes_df

# 3) financials
def process_bronze_users_financials(
    snapshot_date_str,
    bronze_financials_directory,
    spark,
    financials_csv="data/features_financials.csv",
):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    financials_df = spark.read.csv(financials_csv, header=True, inferSchema=True)
    if "snapshot_date" in financials_df.columns:
        financials_df = financials_df.filter(col("snapshot_date") == snapshot_date_str)

    print(snapshot_date_str + " row count (financials):", financials_df.count())

    partition_name = "bronze_users_financials_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_financials_directory + partition_name
    financials_df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return financials_df

# Wrapper with the EXACT original name/signature + same return values
def process_bronze_users(
    snapshot_date_str,
    bronze_clickstream_directory,
    bronze_attributes_directory,
    bronze_financials_directory,
    spark,
    clickstream_csv="data/feature_clickstream.csv",
    attributes_csv="data/features_attributes.csv",
    financials_csv="data/features_financials.csv",
):
    clickstream_df = process_bronze_users_clickstream(
        snapshot_date_str, bronze_clickstream_directory, spark, clickstream_csv
    )
    attributes_df = process_bronze_users_attributes(
        snapshot_date_str, bronze_attributes_directory, spark, attributes_csv
    )
    financials_df = process_bronze_users_financials(
        snapshot_date_str, bronze_financials_directory, spark, financials_csv
    )
    return clickstream_df, attributes_df, financials_df

'''# Users bronze (if clickstream has a date column, pass its name; else leave None)
def process_bronze_users(snapshot_date_str, bronze_clickstream_directory, bronze_attributes_directory, 
                         bronze_financials_directory, spark, clickstream_csv="data/feature_clickstream.csv", 
                         attributes_csv="data/features_attributes.csv", financials_csv="data/features_financials.csv"):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # clickstram
    clickstream_df = (
        spark.read.csv(clickstream_csv, header=True, inferSchema=True)
        .filter(col("snapshot_date") == snapshot_date_str)
    )
    print(snapshot_date_str + " row count (clickstream):", clickstream_df.count())

    partition_name = "bronze_users_clickstream_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_clickstream_directory + partition_name
    clickstream_df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    # attributes
    attributes_df = spark.read.csv(attributes_csv, header=True, inferSchema=True)
    if "snapshot_date" in attributes_df.columns:
        attributes_df = attributes_df.filter(col("snapshot_date") == snapshot_date_str)

    print(snapshot_date_str + " row count (attributes):", attributes_df.count())

    partition_name = "bronze_users_attributes_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_attributes_directory + partition_name
    attributes_df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    # financials
    financials_df = spark.read.csv(financials_csv, header=True, inferSchema=True)
    if "snapshot_date" in financials_df.columns:
        financials_df = financials_df.filter(col("snapshot_date") == snapshot_date_str)

    print(snapshot_date_str + " row count (financials):", financials_df.count())

    partition_name = "bronze_users_financials_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_financials_directory + partition_name
    financials_df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

    return clickstream_df, attributes_df, financials_df'''
