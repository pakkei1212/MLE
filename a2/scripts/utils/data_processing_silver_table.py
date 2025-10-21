import os
import pyspark
import pyspark.sql.functions as F
from datetime import datetime
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# ----------------------------
# Helpers (unchanged)
# ----------------------------
def parse_credit_history_months(col_):
    # "15 Years and 10 Months" -> 190
    years  = F.coalesce(F.regexp_extract(col_, r'(\d+)\s+Years', 1).cast("int"), F.lit(0))
    months = F.coalesce(F.regexp_extract(col_, r'(\d+)\s+Months',1).cast("int"), F.lit(0))
    return years * F.lit(12) + months

def clean_type_of_loan(df):
    # replace " and " with "," -> split -> trim -> dedup -> filter junk -> re-join with "|"
    df = df.withColumn("Type_of_Loan_clean", F.regexp_replace("Type_of_Loan", r"\s+and\s+", ","))
    df = df.withColumn("loan_array", F.split(F.col("Type_of_Loan_clean"), ","))
    df = df.withColumn("loan_array", F.expr("transform(loan_array, x -> trim(x))"))
    df = df.withColumn("loan_array", F.array_distinct("loan_array"))
    df = df.withColumn("loan_array",
                       F.expr("filter(loan_array, x -> x != 'Not Specified' and x != '')"))
    df = df.withColumn("Type_of_Loan", F.array_join("loan_array", "|").cast(StringType()))
    return df.drop("loan_array", "Type_of_Loan_clean")

# ----------------------------
# Silver: LOANS (unchanged)
# ----------------------------
def process_silver_loans(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # read bronze
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + ".csv"
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print("loaded from:", filepath, "row count:", df.count())

    # enforce schema
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for c, t in column_type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    # mob
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # DPD
    df = df.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
    ).fillna(0)
    df = df.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0,
               F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType())
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0,
               F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType())
    )

    # write silver
    out_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + ".parquet"
    out_path = silver_loan_daily_directory + out_name
    df.write.mode("overwrite").parquet(out_path)
    print("saved to:", out_path)
    return df

# ----------------------------
# Silver: ATTRIBUTES
# ----------------------------
def process_silver_users_attributes(snapshot_date_str,
                                    bronze_attributes_directory,
                                    silver_attributes_directory,
                                    spark):
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # read bronze
    part = "bronze_users_attributes_" + snapshot_date_str.replace('-', '_') + ".csv"
    in_path = bronze_attributes_directory + part
    attributes_df = spark.read.csv(in_path, header=True, inferSchema=True)
    print("loaded from:", in_path, "row count:", attributes_df.count())

    # enforce schema
    attr_types = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    for c, t in attr_types.items():
        attributes_df = attributes_df.withColumn(c, col(c).cast(t))

    # cleaning
    attributes_df = attributes_df.withColumn("Name", F.initcap(F.trim(col("Name"))).cast(StringType()))
    attributes_df = attributes_df.withColumn(
        "Name_Masked",
        F.concat(F.substring("Name", 1, 1),
                 F.expr("repeat('*', greatest(length(Name) - 1, 0))")).cast(StringType())
    )

    attributes_df = attributes_df.withColumn("Age", F.regexp_replace("Age", "[^0-9\\-]", ""))
    attributes_df = attributes_df.withColumn(
        "Age",
        F.when((col("Age") < 18) | (col("Age") > 120), F.lit(None))
         .otherwise(col("Age")).cast(IntegerType())
    )

    attributes_df = attributes_df.withColumn("SSN", F.regexp_replace("SSN", "[^0-9\\-]", ""))
    attributes_df = attributes_df.withColumn(
        "SSN",
        F.when(F.col("SSN").rlike("^\\d{3}-\\d{2}-\\d{4}$"), col("SSN"))
         .otherwise(F.lit(None)).cast(StringType())
    )
    attributes_df = attributes_df.withColumn("SSN_Masked",
                                             F.concat(F.lit("XXX-XX-"), F.substring("SSN", -4, 4)))

    attributes_df = attributes_df.withColumn("Occupation",
        F.when(col("Occupation") == "_______", None).otherwise(col("Occupation")))
    attributes_df = attributes_df.withColumn("Occupation",
        F.regexp_replace("Occupation", r"_", " "))
    attributes_df = attributes_df.withColumn("Occupation", col("Occupation").cast(StringType()))

    # write silver
    out = "silver_users_attributes_" + snapshot_date_str.replace('-', '_') + ".parquet"
    out_path = silver_attributes_directory + out
    attributes_df.write.mode("overwrite").parquet(out_path)
    print("saved to:", out_path)
    return attributes_df

# ----------------------------
# Silver: CLICKSTREAM
# ----------------------------
def process_silver_users_clickstream(snapshot_date_str,
                                     bronze_clickstream_directory,
                                     silver_clickstream_directory,
                                     spark):
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # read bronze
    part = "bronze_users_clickstream_" + snapshot_date_str.replace('-', '_') + ".csv"
    in_path = bronze_clickstream_directory + part
    clickstream_df = spark.read.csv(in_path, header=True, inferSchema=True)
    print("loaded from:", in_path, "row count:", clickstream_df.count())

    # enforce schema
    cs_types = {
        **{f"fe_{i}": IntegerType() for i in range(1, 21)},
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    for c, t in cs_types.items():
        clickstream_df = clickstream_df.withColumn(c, col(c).cast(t))

    # write silver
    out = "silver_users_clickstream_" + snapshot_date_str.replace('-', '_') + ".parquet"
    out_path = silver_clickstream_directory + out
    clickstream_df.write.mode("overwrite").parquet(out_path)
    print("saved to:", out_path)
    return clickstream_df

# ----------------------------
# Silver: FINANCIALS
# ----------------------------
def process_silver_users_financials(snapshot_date_str,
                                    bronze_financials_directory,
                                    silver_financials_directory,
                                    spark):
    _ = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # read bronze
    part = "bronze_users_financials_" + snapshot_date_str.replace('-', '_') + ".csv"
    in_path = bronze_financials_directory + part
    financials_df = spark.read.csv(in_path, header=True, inferSchema=True)
    print("loaded from:", in_path, "row count:", financials_df.count())

    # enforce base schema
    fin_types = {
        "Monthly_Inhand_Salary": FloatType(),
        "Delay_from_due_date": IntegerType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Total_EMI_per_month": FloatType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    for c, t in fin_types.items():
        financials_df = financials_df.withColumn(c, col(c).cast(t))

    # numeric cleaning / ranges
    financials_df = financials_df.withColumn(
        "Annual_Income", F.regexp_replace("Annual_Income", "[^0-9.\\-]", "").cast(FloatType())
    )

    financials_df = financials_df.withColumn(
        "Num_Bank_Accounts",
        F.when((col("Num_Bank_Accounts") < 0) | (col("Num_Bank_Accounts") > 50), None)
         .otherwise(col("Num_Bank_Accounts")).cast(IntegerType())
    )

    financials_df = financials_df.withColumn(
        "Num_Credit_Card",
        F.when((col("Num_Credit_Card") < 0) | (col("Num_Credit_Card") > 20), None)
         .otherwise(col("Num_Credit_Card")).cast(IntegerType())
    )

    financials_df = financials_df.withColumn(
        "Interest_Rate",
        F.when((col("Interest_Rate") < 0) | (col("Interest_Rate") > 100), None)
         .otherwise(F.col("Interest_Rate")).cast(FloatType())
    )

    financials_df = financials_df.withColumn(
        "Num_of_Loan", F.regexp_replace("Num_of_Loan", "[^0-9.\\-]", "").cast(IntegerType())
    )
    financials_df = financials_df.withColumn(
        "Num_of_Loan",
        F.when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 20), None)
         .otherwise(col("Num_of_Loan")).cast(IntegerType())
    )

    financials_df = clean_type_of_loan(financials_df)

    financials_df = financials_df.withColumn(
        "Num_Credit_Inquiries",
        F.when((col("Num_Credit_Inquiries") < 0) | (col("Num_Credit_Inquiries") > 50), None)
         .otherwise(col("Num_Credit_Inquiries")).cast(IntegerType())
    )

    financials_df = financials_df.withColumn(
        "Num_of_Delayed_Payment",
        F.regexp_replace("Num_of_Delayed_Payment", "[^0-9.\\-]", "").cast(IntegerType())
    )
    financials_df = financials_df.withColumn(
        "Num_of_Delayed_Payment",
        F.when((col("Num_of_Delayed_Payment") < 0) | (col("Num_of_Delayed_Payment") > 100), None)
         .otherwise(col("Num_of_Delayed_Payment")).cast(IntegerType())
    )

    financials_df = financials_df.withColumn(
        "Changed_Credit_Limit", F.regexp_replace("Changed_Credit_Limit", "[^0-9.\\-]", "").cast(FloatType())
    )

    financials_df = financials_df.withColumn(
        "Credit_Mix",
        F.when(col("Credit_Mix") == "Good", F.lit("Good"))
         .when(col("Credit_Mix") == "Standard", F.lit("Standard"))
         .when(col("Credit_Mix") == "Bad", F.lit("Bad"))
         .otherwise(None).cast(StringType())
    )

    financials_df = financials_df.withColumn(
        "Outstanding_Debt", F.regexp_replace("Outstanding_Debt", "[^0-9.\\-]", "").cast(FloatType())
    )

    financials_df = financials_df.withColumn(
        "Payment_of_Min_Amount",
        F.when(col("Payment_of_Min_Amount").isin("No", "NM"), F.lit("No"))
         .when(col("Payment_of_Min_Amount") == "Yes", F.lit("Yes"))
         .otherwise(None).cast(StringType())
    )

    financials_df = financials_df.withColumn(
        "Amount_invested_monthly", F.regexp_replace("Amount_invested_monthly", "[^0-9.\\-]", "").cast(FloatType())
    )

    # credit history months
    financials_df = financials_df.withColumn(
        "Credit_History_Age", parse_credit_history_months(col("Credit_History_Age")).cast(IntegerType())
    )

    financials_df = financials_df.withColumn(
        "Total_EMI_per_month",
        F.when(col("Total_EMI_per_month") < 0, None)
         .otherwise(col("Total_EMI_per_month")).cast(FloatType())
    )

    # split Payment_Behaviour
    financials_df = financials_df.withColumn(
        "Payment_Behaviour_Spent",
        F.when(col("Payment_Behaviour").rlike("(?i)^Low_spent"), F.lit("Low"))
         .when(col("Payment_Behaviour").rlike("(?i)^High_spent"), F.lit("High"))
         .otherwise(None).cast(StringType())
    )
    financials_df = financials_df.withColumn(
        "Payment_Behaviour_Payment",
        F.when(col("Payment_Behaviour").rlike("(?i)Large_value_payments"),  F.lit("Large"))
         .when(col("Payment_Behaviour").rlike("(?i)Medium_value_payments"), F.lit("Medium"))
         .when(col("Payment_Behaviour").rlike("(?i)Small_value_payments"),  F.lit("Small"))
         .otherwise(None).cast(StringType())
    )
    financials_df = financials_df.drop("Payment_Behaviour")

    # Monthly_Balance numeric
    financials_df = financials_df.withColumn(
        "Monthly_Balance", F.regexp_replace("Monthly_Balance", "[^0-9.\\-]", "").cast(FloatType())
    )

    # write silver
    out = "silver_users_financials_" + snapshot_date_str.replace('-', '_') + ".parquet"
    out_path = silver_financials_directory + out
    financials_df.write.mode("overwrite").parquet(out_path)
    print("saved to:", out_path)
    return financials_df

# ----------------------------
# Wrapper to preserve original signature & return order
# ----------------------------
def process_silver_users(snapshot_date_str,
                         bronze_clickstream_directory, bronze_attributes_directory, bronze_financials_directory,
                         silver_clickstream_directory, silver_attributes_directory, silver_financials_directory,
                         spark):
    attributes_df = process_silver_users_attributes(
        snapshot_date_str,
        bronze_attributes_directory,
        silver_attributes_directory,
        spark,
    )
    clickstream_df = process_silver_users_clickstream(
        snapshot_date_str,
        bronze_clickstream_directory,
        silver_clickstream_directory,
        spark,
    )
    financials_df = process_silver_users_financials(
        snapshot_date_str,
        bronze_financials_directory,
        silver_financials_directory,
        spark,
    )
    # Return in the SAME ORDER as your original function:
    return clickstream_df, attributes_df, financials_df


'''import os
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


def process_silver_loans(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# helper function to parse credit history age
def parse_credit_history_months(col):
        # "15 Years and 10 Months" -> 190
        years  = F.coalesce(F.regexp_extract(col, r'(\d+)\s+Years', 1).cast("int"), F.lit(0))
        months = F.coalesce(F.regexp_extract(col, r'(\d+)\s+Months',1).cast("int"), F.lit(0))
        return years * F.lit(12) + months

# Clean and standardize the Type_of_Loan field
def clean_type_of_loan(df):
    # replace " and " with ","
    df = df.withColumn("Type_of_Loan_clean",
                       F.regexp_replace("Type_of_Loan", r"\s+and\s+", ","))
    # split on commas
    df = df.withColumn("loan_array",
                       F.split(F.col("Type_of_Loan_clean"), ","))
    # trim spaces
    df = df.withColumn("loan_array",
                       F.expr("transform(loan_array, x -> trim(x))"))
    # remove duplicates
    df = df.withColumn("loan_array", F.array_distinct("loan_array"))
    # filter out junk like "Not Specified" or empty
    df = df.withColumn("loan_array",
                       F.expr("filter(loan_array, x -> x != 'Not Specified' and x != '')"))
    # re-join with consistent delimiter
    df = df.withColumn("Type_of_Loan",
                       F.array_join("loan_array", "|").cast(StringType()))

    return df.drop("loan_array", "Type_of_Loan_clean")
        
def process_silver_users(snapshot_date_str, bronze_clickstream_directory, bronze_attributes_directory, 
                         bronze_financials_directory, silver_clickstream_directory, silver_attributes_directory,
                         silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    #============================================
    # attributes
    #============================================   
    # connect to bronze table
    partition_name = "bronze_users_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    attributes_df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', attributes_df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        attributes_df = attributes_df.withColumn(column, col(column).cast(new_type))

    # clean and standardize specific fields
    attributes_df = attributes_df.withColumn("Name", F.initcap(F.trim(col("Name"))).cast(StringType()))
    # Mask Name: keep first letter, mask rest with **
    attributes_df = attributes_df.withColumn(
        "Name_Masked",
        F.concat(
            F.substring("Name", 1, 1),  # keep first character
            F.expr("repeat('*', length(Name) - 1)") # mask rest with *
        ).cast(StringType())
    )
    
    # Age: remove non-numeric characters, convert to integer, set invalid ages to null
    attributes_df = attributes_df.withColumn("Age", F.regexp_replace("Age", "[^0-9\-]", ""))
    attributes_df = attributes_df.withColumn("Age", F.when((col("Age") < 18) | (col("Age") > 120), F.lit(None))
                                             .otherwise(col("Age")).cast(IntegerType()))
    
    # SSN: remove non-numeric and non-dash characters, convert to string, set invalid SSNs to null
    attributes_df = attributes_df.withColumn("SSN", F.regexp_replace("SSN", "[^0-9\-]", ""))
    attributes_df = attributes_df.withColumn("SSN", F.when(F.col("SSN").rlike("^\d{3}-\d{2}-\d{4}$"), 
                                                           col("SSN")).otherwise(F.lit(None)).cast(StringType()))
    
    # Mask SSN: show only last 4 digits, mask rest with XXX-XX-
    attributes_df = attributes_df.withColumn("SSN_Masked", F.concat(F.lit("XXX-XX-"), F.substring("SSN", -4, 4)))
    
    attributes_df = attributes_df.withColumn("Occupation", F.when(col("Occupation") == "_______", None)
                                             .otherwise(col("Occupation")))
    attributes_df = attributes_df.withColumn("Occupation", F.regexp_replace("Occupation", r"_", " "))
    attributes_df = attributes_df.withColumn("Occupation", col("Occupation").cast(StringType()))

    # create Hashed_User_ID as hash of Customer_ID + SSN
    # attributes_df = attributes_df.withColumn("Hashed_User_ID", F.sha2(F.concat_ws("||", col("Customer_ID"), col("SSN")), 256))

    #attributes_df = attributes_df.drop("Name", "SSN")

    
    # save silver table - IRL connect to database to write
    partition_name = "silver_users_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    attributes_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    #============================================
    # clickstream
    #============================================   
    # connect to bronze table
    partition_name = "bronze_users_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    clickstream_df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', clickstream_df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        **{f"fe_{i}": IntegerType() for i in range(1, 21)},
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        clickstream_df = clickstream_df.withColumn(column, col(column).cast(new_type))
 
    # save silver table - IRL connect to database to write
    partition_name = "silver_users_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    clickstream_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    #============================================
    # financials
    #============================================  
    # connect to bronze table
    partition_name = "bronze_users_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    financials_df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', financials_df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Monthly_Inhand_Salary": FloatType(),
        "Delay_from_due_date": IntegerType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Total_EMI_per_month": FloatType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        financials_df = financials_df.withColumn(column, col(column).cast(new_type))

    # clean and standardize specific fields
    financials_df = financials_df.withColumn("Annual_Income", F.regexp_replace("Annual_Income", "[^0-9.\-]", "").cast(FloatType()))

    financials_df = financials_df.withColumn("Num_Bank_Accounts", F.when((col("Num_Bank_Accounts") < 0) | \
                                                                         (col("Num_Bank_Accounts") > 50), None) \
                                                                         .otherwise(col("Num_Bank_Accounts")).cast(IntegerType()))
    
    financials_df = financials_df.withColumn("Num_Credit_Card", F.when((col("Num_Credit_Card") < 0) | \
                                                                       (col("Num_Credit_Card") > 20), None) \
                                                                        .otherwise(col("Num_Credit_Card")).cast(IntegerType()))
    
    financials_df = financials_df.withColumn("Interest_Rate", F.when((col("Interest_Rate") < 0) | \
                                                                     (col("Interest_Rate") > 100), None) \
                                                                     .otherwise(F.col("Interest_Rate")).cast(FloatType()))
    
    financials_df = financials_df.withColumn("Num_of_Loan", F.regexp_replace("Num_of_Loan", "[^0-9.\-]", "").cast(IntegerType()))
    financials_df = financials_df.withColumn("Num_of_Loan", F.when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 20), None)
                                                             .otherwise(col("Num_of_Loan")).cast(IntegerType()))

    financials_df = clean_type_of_loan(financials_df)
    
    financials_df = financials_df.withColumn("Num_Credit_Inquiries", F.when((col("Num_Credit_Inquiries") < 0) | \
                                                                            (col("Num_Credit_Inquiries") > 50), None) \
                                                                            .otherwise(col("Num_Credit_Inquiries")).cast(IntegerType()))

    financials_df = financials_df.withColumn("Num_of_Delayed_Payment", F.regexp_replace("Num_of_Delayed_Payment", "[^0-9.\-]", "").cast(IntegerType()))
    financials_df = financials_df.withColumn("Num_of_Delayed_Payment", F.when((col("Num_of_Delayed_Payment") < 0) | \
                                                                              (col("Num_of_Delayed_Payment") > 100), None)
                                                                              .otherwise(col("Num_of_Delayed_Payment")).cast(IntegerType()))
    
    financials_df = financials_df.withColumn("Changed_Credit_Limit", F.regexp_replace("Changed_Credit_Limit", "[^0-9.\-]", "").cast(FloatType()))

    financials_df = financials_df.withColumn("Credit_Mix", F.when(col("Credit_Mix") == "Good", F.lit("Good")) \
                                                                         .when(col("Credit_Mix") == "Standard", F.lit("Standard")) \
                                                                         .when(col("Credit_Mix") == "Bad", F.lit("Bad")) \
                                                                         .otherwise(None).cast(StringType()))
    
    financials_df = financials_df.withColumn("Outstanding_Debt", F.regexp_replace("Outstanding_Debt", "[^0-9.\-]", "").cast(FloatType()))
    financials_df = financials_df.withColumn("Payment_of_Min_Amount", F.when(col("Payment_of_Min_Amount").isin("No", "NM"), F.lit("No")) \
                                                                         .when(col("Payment_of_Min_Amount") == "Yes", F.lit("Yes")) \
                                                                         .otherwise(None).cast(StringType()))
    
    financials_df = financials_df.withColumn("Amount_invested_monthly", F.regexp_replace("Amount_invested_monthly", "[^0-9.\-]", "").cast(FloatType()))
    
    # parse Credit_History_Age to months (e.g. '16 Years and 3 Months' -> 195)
    financials_df = financials_df.withColumn("Credit_History_Age", parse_credit_history_months(col("Credit_History_Age")).cast(IntegerType()))
    
    financials_df = financials_df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month") < 0, None) \
                                 .otherwise(col("Total_EMI_per_month")).cast(FloatType()))

    # parse Payment_Behaviour into two fields: Spent (Low/High) and Payment (Small/Medium/Large)
    financials_df = financials_df.withColumn("Payment_Behaviour_Spent", F.when(col("Payment_Behaviour").rlike("(?i)^Low_spent"), F.lit("Low")) \
                                                                         .when(col("Payment_Behaviour").rlike("(?i)^High_spent"), F.lit("High")) \
                                                                         .otherwise(None).cast(StringType()))    
    financials_df = financials_df.withColumn("Payment_Behaviour_Payment", F.when(col("Payment_Behaviour").rlike("(?i)Large_value_payments"), F.lit("Large")) \
                                                                         .when(col("Payment_Behaviour").rlike("(?i)Medium_value_payments"), F.lit("Medium")) \
                                                                         .when(col("Payment_Behaviour").rlike("(?i)Small_value_payments"), F.lit("Small")) \
                                                                         .otherwise(None).cast(StringType()))                                                                      

    # drop original Payment_Behaviour column
    financials_df = financials_df.drop("Payment_Behaviour")

    # clean Monthly_Balance: remove non-numeric characters, convert to float
    financials_df = financials_df.withColumn("Monthly_Balance", F.regexp_replace("Monthly_Balance", "[^0-9.\-]", "").cast(FloatType()))

     # select columns to save
    #financials_df = financials_df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
 
    # save silver table - IRL connect to database to write
    partition_name = "silver_users_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    financials_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return clickstream_df, attributes_df, financials_df'''
    
