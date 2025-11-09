# gold_clickstream.py
import argparse, os, pyspark
import utils.data_processing_gold_table as gold

def main(snapshotdate: str):
    print("\n\n--- starting GOLD clickstream job ---\n")
    spark = pyspark.sql.SparkSession.builder.appName("gold-clickstream").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    silver_clickstream_directory = os.path.join("datamart", "silver", "users", "clickstream") + "/"
    gold_clickstream_directory   = os.path.join("datamart", "gold",   "features", "clickstream") + "/"

    os.makedirs(gold_clickstream_directory, exist_ok=True)

    gold.build_gold_feature_clickstream(
        snapshot_date_str=snapshotdate,
        silver_clickstream_directory=silver_clickstream_directory,
        gold_feature_store_directory=gold_clickstream_directory,
        spark=spark,
    )

    spark.stop()
    print("\n--- completed GOLD clickstream job ---\n\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Gold Clickstream Feature Store")
    p.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.snapshotdate)