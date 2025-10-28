# gold_attributes.py
import argparse, os, pyspark
import utils.data_processing_gold_table as gold

def main(snapshotdate: str):
    print("\n\n--- starting GOLD attributes job ---\n")
    spark = pyspark.sql.SparkSession.builder.appName("gold-attributes").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    silver_attributes_directory = os.path.join("datamart", "silver", "users", "attributes") + "/"
    gold_attributes_directory   = os.path.join("datamart", "gold",   "features", "attributes") + "/"

    os.makedirs(gold_attributes_directory, exist_ok=True)

    gold.build_gold_feature_attributes(
        snapshot_date_str=snapshotdate,
        silver_attributes_directory=silver_attributes_directory,
        gold_feature_store_directory=gold_attributes_directory,
        spark=spark,
    )

    spark.stop()
    print("\n--- completed GOLD attributes job ---\n\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Gold Attributes Feature Store")
    p.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.snapshotdate)
