#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, json, glob
from pathlib import Path
from datetime import datetime, date
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None


# ---------------- helpers ----------------
def find_summaries(pred_dir: str):
    # e.g. datamart/gold/model_predictions/**/summary_2024-01-01.json
    paths = glob.glob(str(Path(pred_dir) / "**" / "summary_*.json"), recursive=True)
    paths.sort()
    return paths

def parse_snapshotdate(meta: dict, fallback_from_name: str) -> dt.date | None:
    s = meta.get("snapshotdate") or meta.get("snapshot_date") or meta.get("evaluation_time")
    if s:
        try:
            return dt.date.fromisoformat(str(s))
        except Exception:
            pass
    import re
    m = re.search(r"summary_(\d{4}-\d{2}-\d{2})\.json$", fallback_from_name)
    if m:
        return dt.date.fromisoformat(m.group(1))
    return None

def _coerce_date(x):
    if isinstance(x, (datetime, date)):
        return x if isinstance(x, date) else x.date()
    return pd.to_datetime(x).date()

def _month_floor(d: date) -> date:
    return date(d.year, d.month, 1)

def _range_months(end_date: date, window_months: int):
    months, cur = [], _month_floor(end_date)
    for _ in range(window_months):
        months.append(cur)
        y = cur.year - (1 if cur.month == 1 else 0)
        m = 12 if cur.month == 1 else cur.month - 1
        cur = date(y, m, 1)
    months.reverse()
    return months

def _psi(reference, current, bins=10):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if reference.size < 50 or current.size < 50:
        return None
    try:
        qs = np.linspace(0, 1, bins + 1)
        cuts = np.unique(np.quantile(reference, qs))
        if len(cuts) < 3:
            return None
        r_hist, _ = np.histogram(reference, bins=cuts)
        c_hist, _ = np.histogram(current,   bins=cuts)
        r_prop = (r_hist + 1e-6) / (r_hist.sum() + 1e-6 * len(r_hist))
        c_prop = (c_hist + 1e-6) / (c_hist.sum() + 1e-6 * len(c_hist))
        return float(np.sum((c_prop - r_prop) * np.log(c_prop / r_prop)))
    except Exception:
        return None

def _safe_ks(ref, cur):
    if ks_2samp is None or len(ref) < 2 or len(cur) < 2:
        return None
    try:
        return float(ks_2samp(ref, cur).statistic)
    except Exception:
        return None

def _month_floor_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: _month_floor(x))


# ---------------- IO loaders ----------------
def iter_summary_json(predictions_dir: str):
    """Yield (snapshot_date:date, model_selector, model_uri, summary_dict)."""
    for summary in Path(predictions_dir).glob("**/summary_*.json"):
        try:
            with open(summary, "r") as f:
                data = json.load(f)
            snap_date = parse_snapshotdate(data, summary.name)
            if not snap_date:
                continue
            selector = data.get("model_selector")
            uri = data.get("model_uri")
            yield snap_date, selector, uri, data
        except Exception:
            continue

def load_scores_for_month(predictions_dir: str, month: date) -> np.ndarray:
    """Read scores from parquet for a given month if available."""
    root = Path(predictions_dir)
    scores = []
    month_tag = f"{month.year}_{month.month:02d}_01"

    # try any parquet that matches the month in its filename OR load and filter by a date column
    parqs = list(root.glob(f"**/*{month_tag}*.parquet")) + list(root.glob("**/*.parquet"))
    for p in parqs:
        try:
            df = pd.read_parquet(p)
            # detect score & date columns
            score_col = next((c for c in [
                "predicted_default_risk", "score", "prob", "proba"
            ] if c in df.columns), None)
            if score_col is None:
                continue

            date_col = next((c for c in [
                "snapshot_date", "label_snapshot_date", "date"
            ] if c in df.columns), None)
            if date_col is not None:
                d = pd.to_datetime(df[date_col], errors="coerce").dt.date
                df = df[_month_floor_series(d) == month]

            vals = pd.to_numeric(df[score_col], errors="coerce").dropna().values
            if len(vals):
                scores.append(vals)
        except Exception:
            continue
    return np.concatenate(scores) if scores else np.array([])


# ---------------- plotting ----------------
def _maybe_plot(monthly: pd.DataFrame, out_png: Path, include_drift: bool):
    try:
        plt.figure(figsize=(10, 6))

        # x axis
        x = pd.to_datetime(monthly["month"])
        label_x = x.dt.strftime("%Y-%m")

        # plot what we have
        if "sample_size" in monthly:
            plt.plot(label_x, monthly["sample_size"], marker="o", label="sample_size")
        if "accuracy" in monthly:
            plt.plot(label_x, monthly["accuracy"], marker="o", label="accuracy")
        if "f1_macro" in monthly:
            plt.plot(label_x, monthly["f1_macro"], marker="o", label="f1_macro")
        if "auc_roc" in monthly:
            plt.plot(label_x, monthly["auc_roc"], marker="o", label="auc_roc")
        if include_drift and "psi_vs_trailing" in monthly:
            plt.plot(label_x, monthly["psi_vs_trailing"], marker="o", label="psi_vs_trailing")
        if include_drift and "ks_vs_trailing" in monthly:
            plt.plot(label_x, monthly["ks_vs_trailing"], marker="o", label="ks_vs_trailing")

        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)
        plt.title("Monitoring metrics over time (monthly)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"[MONITOR] Saved plot: {out_png}")
    except Exception as e:
        print(f"[MONITOR] Plotting skipped due to error: {e}")


# --------------- main logic ----------------
def main(args):
    # Use schedule-provided end date if given (e.g. Airflow {{ ds }}), else today
    end_date = _coerce_date(args.end_date) if args.end_date else _coerce_date(datetime.utcnow().date())
    months = _range_months(end_date=end_date, window_months=args.window_months)
    window_start = months[0]

    # 1) collect monthly rows from summaries
    rows = []
    for snap_date, selector, uri, info in iter_summary_json(args.predictions_dir):
        m = _month_floor(snap_date)
        if m not in months:
            continue

        # pick up either your old field names OR the new ones from model_inference.py
        score_min = info.get("score_min") or info.get("predicted_default_risk_min")
        score_max = info.get("score_max") or info.get("predicted_default_risk_max")
        score_mean = info.get("score_mean") or info.get("predicted_default_risk_mean")
        score_std = info.get("score_std") or info.get("score_stdev")

        rows.append({
            "month": m,
            "model_selector": selector,
            "model_uri": uri,
            "sample_size": info.get("sample size") or info.get("sample_size") or info.get("n_scored"),
            "score_min": score_min,
            "score_max": score_max,
            "score_mean": score_mean,
            "score_std": score_std,
            "accuracy": info.get("accuracy"),
            "precision": info.get("precision"),
            "recall": info.get("recall"),
            "f1_macro": info.get("macro_f1") or info.get("f1_macro"),
            "f1_weighted": info.get("weight_f1") or info.get("f1_weighted"),
            "auc_roc": info.get("auc_roc") or info.get("auc_roc"),
        })

    if not rows:
        print("[MONITOR] No prediction summaries found in the requested window.")
        return 0

    monthly = (
        pd.DataFrame(rows)
        .sort_values(["month", "sample_size"], ascending=[True, False])
        .drop_duplicates(subset=["month"], keep="first")
        .sort_values("month")
        .reset_index(drop=True)
    )

    # 2) optional drift metrics: compare each month to trailing reference
    if args.enable_drift:
        trailing_scores = {}
        ks_vals, psi_vals = [], []
        month_set = set(monthly["month"])

        for i, m in enumerate(months):
            if m not in month_set:
                ks_vals.append(None); psi_vals.append(None); continue

            cur_scores = load_scores_for_month(args.predictions_dir, m)

            if i > 0:
                ref_parts = []
                for j in range(max(0, i-args.psi_trailing), i):
                    mj = months[j]
                    if mj in trailing_scores and trailing_scores[mj].size:
                        ref_parts.append(trailing_scores[mj])
                ref = np.concatenate(ref_parts) if ref_parts else np.array([])
            else:
                ref = np.array([])

            ks_vals.append(_safe_ks(ref, cur_scores) if ref.size and cur_scores.size else None)
            psi_vals.append(_psi(ref, cur_scores) if ref.size and cur_scores.size else None)
            trailing_scores[m] = cur_scores

        drift_map = {m: (ks, psi) for m, ks, psi in zip(months, ks_vals, psi_vals)}
        monthly["ks_vs_trailing"] = monthly["month"].map(lambda mm: drift_map.get(mm, (None, None))[0])
        monthly["psi_vs_trailing"] = monthly["month"].map(lambda mm: drift_map.get(mm, (None, None))[1])

    # 3) write outputs
    out_root = Path(args.out_dir)
    (out_root / "monthly").mkdir(parents=True, exist_ok=True)

    # per-month files
    for _, r in monthly.iterrows():
        tag = f"{r['month'].year}_{r['month'].month:02d}_01"
        pd.DataFrame([r]).to_parquet(out_root / "monthly" / f"monitor_{tag}.parquet", index=False)

    # consolidated parquet for the window
    monthly_out = out_root / f"monitor_window_{window_start.isoformat()}_to_{end_date.isoformat()}.parquet"
    monthly_to_save = monthly.copy()
    monthly_to_save["month"] = pd.to_datetime(monthly_to_save["month"])
    monthly_to_save.to_parquet(monthly_out, index=False)

    # optional plot
    if args.make_plots:
        plot_path = out_root / f"monitor_trends_{window_start.isoformat()}_to_{end_date.isoformat()}.png"
        _maybe_plot(monthly, plot_path, include_drift=args.enable_drift)

    # quick recap JSON (+ simple AUC health flag)
    auc_series = pd.to_numeric(monthly.get("auc_roc"), errors="coerce").dropna()
    auc_avg = float(auc_series.mean()) if len(auc_series) else None
    auc_flag = None
    if auc_avg is not None:
        # rough guidance: >=0.70 good; 0.65–0.70 caution; <0.65 investigate
        if auc_avg >= 0.70: auc_flag = "good"
        elif auc_avg >= 0.65: auc_flag = "caution"
        else: auc_flag = "investigate"

    quick = {
        "window_months": int(args.window_months),
        "window_start": str(window_start),
        "end_date": str(end_date),
        "coverage_months": int(monthly["month"].nunique()),
        "avg_score_mean": float(pd.to_numeric(monthly.get("score_mean"), errors="coerce").dropna().mean()) if "score_mean" in monthly else None,
        "avg_accuracy": float(pd.to_numeric(monthly.get("accuracy"), errors="coerce").dropna().mean()) if "accuracy" in monthly else None,
        "avg_f1_macro": float(pd.to_numeric(monthly.get("f1_macro"), errors="coerce").dropna().mean()) if "f1_macro" in monthly else None,
        "avg_auc": auc_avg,
        "auc_health": auc_flag,
        "avg_psi_vs_trailing": float(pd.to_numeric(monthly.get("psi_vs_trailing", pd.Series(dtype=float)), errors="coerce").dropna().mean()) if args.enable_drift else None,
        "avg_ks_vs_trailing": float(pd.to_numeric(monthly.get("ks_vs_trailing", pd.Series(dtype=float)), errors="coerce").dropna().mean()) if args.enable_drift else None,
        "paths": {
            "monthly_dir": str((out_root / "monthly").resolve()),
            "consolidated_parquet": str(monthly_out.resolve()),
            "trend_plot": str((out_root / f"monitor_trends_{window_start.isoformat()}_to_{end_date.isoformat()}.png").resolve()) if args.make_plots else None
        }
    }
    with open(out_root / f"monitor_last_{args.window_months}m_end_{end_date}.json", "w") as f:
        json.dump(quick, f, indent=2)

    print("[MONITOR] Saved consolidated:", monthly_out)
    print("[MONITOR] Monthly dir:", (out_root / "monthly").resolve())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Label-free model monitoring from prediction summaries (+optional drift).")
    ap.add_argument("--predictions-dir", type=str, default="datamart/gold/model_predictions/")
    ap.add_argument("--out-dir", type=str, default="datamart/gold/model_monitoring/")
    ap.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--window-months", type=int, default=12)
    ap.add_argument("--enable-drift", action="store_true", help="If set, compute KS/PSI using predictions parquet.")
    ap.add_argument("--psi-trailing", type=int, default=3, help="Trailing months to build reference for KS/PSI.")
    ap.add_argument("--make-plots", action="store_true", help="If set, saves a PNG trend chart of key metrics.")
    sys.exit(main(ap.parse_args()))
