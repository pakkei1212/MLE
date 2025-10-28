# gold_features.py
import argparse
import os
import pyspark

import utils.data_processing_pretrain_gold_table as pretrain


def main(snapshotdate: str, mob: int, dedupe: str, drop_nulls: bool):
    print("\n\n--- starting GOLD pretrain (from gold sources) ---\n")

    # Spark
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("gold-pretrain-from-gold")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Paths (adjust if your layout differs)
    gold_label_store_directory            = os.path.join("datamart", "gold", "labels") + "/"
    gold_feature_attributes_directory     = os.path.join("datamart", "gold", "features", "attributes") + "/"
    gold_feature_clickstream_directory    = os.path.join("datamart", "gold", "features", "clickstream") + "/"
    gold_feature_financials_directory     = os.path.join("datamart", "gold", "features", "financials") + "/"
    gold_pretrain_feature_store_directory = os.path.join("datamart", "pretrain_gold", "features") + "/"

    # Ensure output dir exists
    os.makedirs(gold_pretrain_feature_store_directory, exist_ok=True)

    # Build pretrain features (consumes gold_* sources + gold labels)
    pretrain.build_gold_pretrain_feature_store_from_gold(
        snapshot_date_str=snapshotdate,
        gold_label_store_directory=gold_label_store_directory,
        gold_feature_attributes_directory=gold_feature_attributes_directory,
        gold_feature_clickstream_directory=gold_feature_clickstream_directory,
        gold_feature_financials_directory=gold_feature_financials_directory,
        gold_pretrain_feature_store_directory=gold_pretrain_feature_store_directory,
        spark=spark,
        mob=mob,
        dedupe_strategy=dedupe,           # "latest" or "none"
        drop_null_feature_rows=drop_nulls # optional cleanup
    )

    spark.stop()
    print("\n--- completed GOLD pretrain (from gold sources) ---\n\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build Pretrain Feature Store from Gold sources")
    p.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (label snapshot date)")
    p.add_argument("--mob", type=int, required=True, help="Months-on-books for cutoff (e.g., 6)")
    p.add_argument("--dedupe", type=str, default="latest", choices=["latest", "none"],
                   help="Deduplicate to one row per Customer_ID (prefer freshest snapshots) or keep all")
    p.add_argument("--drop-nulls", action="store_true",
                   help="Drop rows with NULLs in feature columns (keeps metadata)")
    args = p.parse_args()

    main(args.snapshotdate, args.mob, args.dedupe, args.drop_nulls)
