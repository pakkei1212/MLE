# gold_financials.py
import argparse, os, pyspark
import utils.data_processing_gold_table as gold

def main(snapshotdate: str):
    print("\n\n--- starting GOLD financials job ---\n")
    spark = pyspark.sql.SparkSession.builder.appName("gold-financials").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    silver_financials_directory = os.path.join("datamart", "silver", "users", "financials") + "/"
    gold_financials_directory   = os.path.join("datamart", "gold",   "features", "financials") + "/"

    os.makedirs(gold_financials_directory, exist_ok=True)

    gold.build_gold_feature_financials(
        snapshot_date_str=snapshotdate,
        silver_financials_directory=silver_financials_directory,
        gold_feature_store_directory=gold_financials_directory,
        spark=spark,
    )

    spark.stop()
    print("\n--- completed GOLD financials job ---\n\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Gold Financials Feature Store")
    p.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.snapshotdate)
