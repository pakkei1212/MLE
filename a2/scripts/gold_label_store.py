# gold_label_store.py
import argparse, os, pyspark
import utils.data_processing_gold_table as gold

# Usage: python gold_label_store.py --snapshotdate "2023-01-01" --dpd 30 --mob 6
def main(snapshotdate: str, dpd: int, mob: int):
    print("\n\n---starting GOLD label store job---\n")
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    silver_loan_daily_directory = os.path.join("datamart", "silver", "lms") + "/"
    gold_label_store_directory  = os.path.join("datamart", "gold",  "labels") + "/"
    os.makedirs(gold_label_store_directory, exist_ok=True)

    gold.build_gold_label_store(
        snapshotdate,
        silver_loan_daily_directory,
        gold_label_store_directory,
        spark,
        dpd=dpd,
        mob=mob,
    )

    spark.stop()
    print("\n---completed GOLD label store job---\n\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Gold Label Store")
    p.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--dpd", type=int, required=True, help="DPD threshold (e.g. 30)")
    p.add_argument("--mob", type=int, required=True, help="Months on book (e.g. 6)")
    args = p.parse_args()
    main(args.snapshotdate, args.dpd, args.mob)