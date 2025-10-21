# gold_feature_store.py
import argparse, os, pyspark
import utils.data_processing_gold_table as gold

# Usage: python gold_feature_store.py --snapshotdate "2023-01-01" --mob 6
def main(snapshotdate: str, mob: int):
    print("\n\n---starting GOLD feature store job---\n")
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    gold_label_store_directory   = os.path.join("datamart", "gold",  "labels") + "/"
    silver_clickstream_directory = os.path.join("datamart", "silver", "users", "clickstream") + "/"
    silver_attributes_directory  = os.path.join("datamart", "silver", "users", "attributes") + "/"
    silver_financials_directory  = os.path.join("datamart", "silver", "users", "financials") + "/"
    gold_feature_store_directory = os.path.join("datamart", "gold",  "features") + "/"

    os.makedirs(gold_feature_store_directory, exist_ok=True)

    gold.build_gold_feature_store(
        snapshotdate,
        gold_label_store_directory,
        silver_clickstream_directory,
        silver_attributes_directory,
        silver_financials_directory,
        gold_feature_store_directory,
        spark,
        mob=mob,
    )

    spark.stop()
    print("\n---completed GOLD feature store job---\n\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Gold Feature Store")
    p.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--mob", type=int, required=True, help="Months on book used for pre-label window (e.g. 6)")
    args = p.parse_args()
    main(args.snapshotdate, args.mob)
