# scripts/promote_best.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os, io, sys, json
from datetime import datetime

import numpy as np
import pandas as pd

from utils.model_training_utils import (
    spark_session,
    write_model_registry_row,
    maybe_promote_to_production,
)

# --- robust loader (handles custom transformers moved out of __main__) ---
def _load_artefact(path: str):
    import pickle, cloudpickle

    scripts_dir = "/opt/airflow/scripts"
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from ml_transforms import LoanTypeBinarizer  # noqa: F401
    except Exception:
        pass

    class RedirectUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name in {"LoanTypeBinarizer"}:
                module = "ml_transforms"
            return super().find_class(module, name)

    with open(path, "rb") as f:
        data = f.read()
    try:
        return cloudpickle.loads(data)
    except Exception:
        return RedirectUnpickler(io.BytesIO(data)).load()


def main(args):
    # Inputs
    train_date = args.train_date  # "YYYY-MM-DD"
    date_token = train_date.replace("-", "_")  # used in model_version suffixes

    # Load registry parquet
    spark = spark_session()
    try:
        reg_sdf = spark.read.parquet(args.registry_dir)
    except Exception as e:
        print(f"[PROMOTE] Registry not found or unreadable at {args.registry_dir}: {e}")
        return

    reg_pdf = reg_sdf.toPandas()
    if reg_pdf.empty:
        print("[PROMOTE] Registry is empty.")
        return

    # Candidate rows: same training date only (match by model_version suffix)
    # Expected versions like: credit_model_xgb_2024_01_01, credit_model_logreg_2024_01_01, credit_model_lgbm_2024_01_01
    mask = reg_pdf["model_version"].astype(str).str.endswith(date_token)
    cands = reg_pdf.loc[mask].copy()
    if cands.empty:
        print(f"[PROMOTE] No registry entries for train_date={train_date} (token={date_token}).")
        return

    # Best by auc_oot desc, tie-break by auc_test desc
    cands["__auc_oot__"] = pd.to_numeric(cands["auc_oot"], errors="coerce")
    cands["__auc_test__"] = pd.to_numeric(cands["auc_test"], errors="coerce")
    cands = cands.sort_values(["__auc_oot__", "__auc_test__"], ascending=[False, False])
    best_row = cands.iloc[0]
    model_version = best_row["model_version"]
    print(f"[PROMOTE] Candidate best (same-day): {model_version} | auc_oot={best_row['auc_oot']:.4f} | auc_test={best_row['auc_test']:.4f}")

    # Load artefact for that version
    model_dir = os.path.join(args.model_bank_dir, model_version)
    artefact_path = os.path.join(model_dir, model_version + ".pkl")
    if not os.path.exists(artefact_path):
        print(f"[PROMOTE] Artefact not found: {artefact_path}")
        return
    artefact = _load_artefact(artefact_path)

    # Promote if strictly better than current production (uses your helper)
    promoted = maybe_promote_to_production(
        artefact,
        version_str=model_version,
        production_dir=args.production_dir,
        epsilon=args.epsilon,
    )
    print(f"[PROMOTE] promoted={promoted}")

    # Append a fresh registry row reflecting this promotion attempt (and keep the same windows/metrics)
    cfg = artefact.get("config", {})
    promoted_flag = bool(promoted)
    promoted_at_iso = datetime.now().astimezone().isoformat() if promoted_flag else None

    # windows may be strings already in cfg (we accept both)
    train_start = cfg.get("train_test_start_date")
    train_end   = cfg.get("train_test_end_date")
    oot_start   = cfg.get("oot_start_date")
    oot_end     = cfg.get("oot_end_date")

    res = artefact.get("results", {})
    auc_train = float(res.get("auc_train", np.nan))
    auc_test  = float(res.get("auc_test",  np.nan))
    auc_oot   = float(res.get("auc_oot",   np.nan))

    write_model_registry_row(
        spark,
        registry_dir=args.registry_dir,
        model_version=model_version,
        train_start=train_start, train_end=train_end,
        oot_start=oot_start,     oot_end=oot_end,
        auc_train=auc_train, auc_test=auc_test, auc_oot=auc_oot,
        promoted_flag=promoted_flag,
        promoted_at_iso=promoted_at_iso
    )
    print(f"[PROMOTE] Wrote registry row (promoted_flag={promoted_flag}) to {args.registry_dir}")

    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Promote the best same-day model from Parquet registry to production.")
    ap.add_argument("--train-date", required=True, help="YYYY-MM-DD (choose among models trained on this date only)")
    ap.add_argument("--registry-dir", default=os.path.join("datamart", "gold", "model_registry"), help="Parquet registry dir")
    ap.add_argument("--model-bank-dir", default="model_bank", help="Where artefacts live (per-version subfolders)")
    ap.add_argument("--production-dir", default=os.path.join("model_bank", "production", "best"), help="Production/best folder")
    ap.add_argument("--epsilon", type=float, default=1e-6, help="Min OOT AUC improvement over current prod to promote")
    args = ap.parse_args()
    main(args)
