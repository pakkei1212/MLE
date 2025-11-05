#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from utils.model_training_utils import (
    spark_session,
    maybe_promote_to_production,
)


def main(args):
    # ------------------------------------------------------------------
    # 0. MLflow connection setup
    # ------------------------------------------------------------------
    tracking_uri = (
        args.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or "http://mlflow:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    print(f"[PROMOTE] Using MLflow tracking URI: {tracking_uri}")
    print(f"[PROMOTE] Experiment: {args.experiment}")
    print(f"[PROMOTE] Model name: {args.model_name}")
    print(f"[PROMOTE] Target stage: {args.stage} | archive_existing={args.archive_existing}")

    train_date = args.train_date  # e.g. "2024-09-01"

    # ------------------------------------------------------------------
    # 1. Search MLflow runs with the given train_date tag
    # ------------------------------------------------------------------
    try:
        exp = client.get_experiment_by_name(args.experiment)
        if not exp:
            print(f"[PROMOTE] Experiment '{args.experiment}' not found.")
            return
        exp_id = exp.experiment_id
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.train_date = '{train_date}'",
            max_results=500,
        )
    except Exception as e:
        print(f"[PROMOTE] Failed to query MLflow runs: {e}")
        return

    if not runs:
        print(f"[PROMOTE] No MLflow runs found for train_date={train_date}.")
        return

    df = pd.DataFrame(
        [
            {
                "run_id": r.info.run_id,
                "auc_train": r.data.metrics.get("auc_train"),
                "auc_test": r.data.metrics.get("auc_test"),
                "auc_oot": r.data.metrics.get("auc_oot"),
                "train_test_start_date": r.data.params.get("train_test_start_date"),
                "train_test_end_date": r.data.params.get("train_test_end_date"),
                "oot_start_date": r.data.params.get("oot_start_date"),
                "oot_end_date": r.data.params.get("oot_end_date"),
            }
            for r in runs
        ]
    )

    df["auc_oot"] = pd.to_numeric(df["auc_oot"], errors="coerce")
    df["auc_test"] = pd.to_numeric(df["auc_test"], errors="coerce")

    # ------------------------------------------------------------------
    # 2. Match MLflow model versions by run_id
    # ------------------------------------------------------------------
    versions = client.search_model_versions(f"name='{args.model_name}'")
    run_to_version = {v.run_id: v.version for v in versions}
    df = df[df["run_id"].isin(run_to_version.keys())]

    if df.empty:
        print(f"[PROMOTE] No registered versions found for {args.model_name} with train_date={train_date}.")
        return

    df["model_version"] = df["run_id"].map(run_to_version)

    # Print top 3 for transparency
    print("\n[TOP 3 CANDIDATES]")
    print(df.sort_values(["auc_oot", "auc_test"], ascending=[False, False]).head(3)[
        ["model_version", "run_id", "auc_train", "auc_test", "auc_oot"]
    ])

    # ------------------------------------------------------------------
    # 3. Select the best model by AUC_OOT (and AUC_TEST as tiebreaker)
    # ------------------------------------------------------------------
    df = df.sort_values(["auc_oot", "auc_test"], ascending=[False, False])
    best = df.iloc[0]
    best_run_id = best["run_id"]
    best_ver = str(best["model_version"])
    print(
        f"\n[PROMOTE] Best run for train_date={train_date}: "
        f"version={best_ver}, run_id={best_run_id}, "
        f"auc_oot={best['auc_oot']:.4f}, auc_test={best['auc_test']:.4f}"
    )

    # ------------------------------------------------------------------
    # 4. Build artefact-like dict for MLflow promotion
    # ------------------------------------------------------------------
    artefact = {
        "results": {
            "auc_train": best["auc_train"],
            "auc_test": best["auc_test"],
            "auc_oot": best["auc_oot"],
        },
        "config": {
            "train_test_start_date": best["train_test_start_date"],
            "train_test_end_date": best["train_test_end_date"],
            "oot_start_date": best["oot_start_date"],
            "oot_end_date": best["oot_end_date"],
        },
    }

    # ------------------------------------------------------------------
    # 5. Promote the best model in MLflow registry
    # ------------------------------------------------------------------
    promoted = maybe_promote_to_production(
        artefact=artefact,
        version_str=best_ver,
        model_name=args.model_name,
        epsilon=args.epsilon,
    )

    if promoted:
        # Transition to user-specified stage (default: Production)
        client.transition_model_version_stage(
            name=args.model_name,
            version=best_ver,
            stage=args.stage,
            archive_existing_versions=bool(args.archive_existing),
        )
        print(f"[PROMOTE] ✅ Model version {best_ver} moved to stage '{args.stage}'.")
    else:
        print(f"[PROMOTE] ⚠️ Promotion skipped — not better than existing {args.stage} model.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Promote the best same-day MLflow model (tag=train_date) to specified stage."
    )
    ap.add_argument("--train-date", required=True, help="YYYY-MM-DD (match MLflow tag 'train_date')")
    ap.add_argument("--tracking-uri", default=None, help="Fallback to $MLFLOW_TRACKING_URI or http://mlflow:5000")
    ap.add_argument("--experiment", default="credit_risk_training")
    ap.add_argument("--model-name", default="credit_risk_model")
    ap.add_argument("--stage", default="Production", choices=["Staging", "Production", "Archived", "None"])
    ap.add_argument("--archive-existing", type=int, default=1, help="1/0 to archive existing versions in target stage")
    ap.add_argument("--epsilon", type=float, default=1e-6, help="Minimum OOT AUC improvement for promotion")
    args = ap.parse_args()
    main(args)
