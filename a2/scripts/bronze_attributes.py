import argparse
import os
import pyspark
from pyspark.sql.functions import col

import utils.data_processing_bronze_table  # expects process_bronze_attributes()

# Usage: python bronze_attributes.py --snapshotdate "2023-01-01"
def main(snapshotdate):
    print('\n\n---starting attributes job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    bronze_attributes_directory = os.path.join("datamart", "bronze", "users", "attributes") + "/"
    os.makedirs(bronze_attributes_directory, exist_ok=True)

    utils.data_processing_bronze_table.process_bronze_attributes(
        date_str, bronze_attributes_directory, spark
        # , attributes_csv="data/features_attributes.csv"  # optional override
    )

    spark.stop()
    print('\n\n---completed attributes job---\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run attributes bronze job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
