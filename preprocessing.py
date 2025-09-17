from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pprint import pprint


def selecting_feature_set(df):
    print("----- 1. Selecting Final Feature Set -----")
    selected_features = [
        " Timestamp",

        # Traffic volume & throughput
        " Flow Duration",
        " Total Fwd Packets",
        " Total Backward Packets",
        "Flow Bytes/s",
        " Flow Packets/s",
        " Average Packet Size",

        # Packet size stats
        " Packet Length Mean",
        " Packet Length Std",
        " Fwd Packet Length Mean",
        " Bwd Packet Length Mean",

        # Timing / burstiness 
        " Flow IAT Mean",
        " Flow IAT Std",
        " Fwd IAT Mean",
        " Bwd IAT Mean",
        "Active Mean",
        "Idle Mean",

        # Asymmetry 
        " Down/Up Ratio",
        " Avg Fwd Segment Size",
        " Avg Bwd Segment Size",

        # TCP flag behavior 
        " SYN Flag Count",
        " ACK Flag Count",
        " RST Flag Count",

        # Protocol context 
        " Protocol",
        " Destination Port",

        # Label for anomaly detection
        " Label"
    ]
    
    df = df[selected_features]
    return df


def missing_summary(df):
    # Column-wise summary 
    col_summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().mean() * 100).round(2)
    })
    col_summary = col_summary[col_summary["missing_count"] > 0]
    col_summary = col_summary.sort_values("missing_count", ascending=False)

    # Row-wise summary (just total count)
    row_missing_count = (df.isnull().sum(axis=1) > 0).sum()

    print("Missing by Column:")
    if col_summary.empty:
        print("No missing values in any column.\n")
    else:
        print(col_summary, "\n")

    print("Missing by Row:")
    if row_missing_count == 0:
        print("No rows with missing values.")
    else:
        print(f"{row_missing_count} rows contain at least one missing value.")

    return col_summary, row_missing_count


def handling_missing_vals(
    df,
    mode="train",
    high_thresh=0.80,   # ≥80% missing → drop (unless protected)
    ffill_limit=6,      # limit for forward-fill
    protected_cols=(" Protocol", " Destination Port", " Label"),
    return_report=True
):
    print("----- 2. Handling Missing Values In Data -----")
    print("----- Missing Report -----")
    df = df.copy()
    col_summary, row_summary = missing_summary(df)
    print(col_summary)
    print(row_summary)

    # column groups 
    interp_cols = [
        " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
        "Flow Bytes/s", " Flow Packets/s", " Average Packet Size",
        " Packet Length Mean", " Packet Length Std",
        " Fwd Packet Length Mean", " Bwd Packet Length Mean",
        " Down/Up Ratio", " Avg Fwd Segment Size", " Avg Bwd Segment Size"
    ]
    ffill_cols = [
        " Flow IAT Mean", " Flow IAT Std", " Fwd IAT Mean", " Bwd IAT Mean",
        "Active Mean", "Idle Mean",
        " SYN Flag Count", " ACK Flag Count", " RST Flag Count",
        " Protocol", " Destination Port"
    ]
    categoricals = [" Protocol", " Destination Port"]

    # filter only existing columns
    interp_cols = [c for c in interp_cols if c in df.columns]
    ffill_cols  = [c for c in ffill_cols  if c in df.columns]
    categoricals = [c for c in categoricals if c in df.columns]

    # 1) check missingness 
    miss_frac = df.isna().mean().to_dict()

    # 2) drop heavily missing columns (except protected)
    drop_cols = [col for col, frac in miss_frac.items() if frac >= high_thresh and col not in protected_cols]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        interp_cols = [c for c in interp_cols if c in df.columns]
        ffill_cols  = [c for c in ffill_cols  if c in df.columns]
        categoricals = [c for c in categoricals if c in df.columns]

    # 3) numeric features: ffill → interpolate → median fallback
    if interp_cols:
        df[interp_cols] = df[interp_cols].ffill(limit=ffill_limit)
        df[interp_cols] = df[interp_cols].interpolate(method="linear", limit=ffill_limit, limit_direction="both")
        medians = df[interp_cols].median(numeric_only=True)
        df[interp_cols] = df[interp_cols].fillna(medians)

    # 4) burstiness/flags/categoricals
    for col in ffill_cols:
        df[col] = df[col].ffill(limit=ffill_limit)
        if df[col].isna().any():
            try:
                mode_val = df[col].mode(dropna=True).iloc[0]
                df[col] = df[col].fillna(mode_val)
            except Exception:
                df[col] = df[col].fillna("Unknown" if col in categoricals else 0)

    # 5) label handling 
    if " Label" in df.columns:
        if mode == "train":
            df = df.dropna(subset=[" Label"])
        elif mode == "inference":
            df[" Label"] = df[" Label"].fillna("Unknown")
        else:
            raise ValueError("mode must be 'train' or 'inference'")

    report = {
        "dropped_columns": drop_cols,
        "missing_fraction": {k: float(v) for k, v in miss_frac.items()},
        "params": {
            "mode": mode,
            "high_thresh": high_thresh,
            "ffill_limit": ffill_limit,
            "protected_cols": list(protected_cols),
        },
    }

    return (df, report) if return_report else df

    
def preprocessing(csv_path, interval="5S"):
    # Collect all CSV files in the folder
    all_files = [f for f in listdir(csv_path) if isfile(join(csv_path, f))]
    dfs = []

    for file in all_files:
        file_path = join(csv_path, file)
        print(f"[LOAD] {file_path}")
        temp_df = pd.read_csv(file_path)
        temp_df = selecting_feature_set(temp_df)
        temp_df, report = handling_missing_vals(temp_df)
        
        pprint(report, sort_dicts=False, width=100)

        # Parse timestamp
        temp_df[" Timestamp"] = pd.to_datetime(temp_df[" Timestamp"], errors="coerce")
        """temp_df = temp_df.dropna(subset=[" Timestamp"]).set_index(" Timestamp").sort_index()

        # --- Resample into fixed intervals ---
        ts = temp_df.resample(interval).agg({
            " Total Fwd Packets": "sum",
            " Total Backward Packets": "sum",
            "Flow Bytes/s": "mean",
            " Flow Packets/s": "mean",
            " Average Packet Size": "mean",
            " Packet Length Mean": "mean",
            " Packet Length Std": "mean",
            " Fwd Packet Length Mean": "mean",
            " Bwd Packet Length Mean": "mean",
            " Flow IAT Mean": "mean",
            " Flow IAT Std": "mean",
            " Fwd IAT Mean": "mean",
            " Bwd IAT Mean": "mean",
            "Active Mean": "mean",
            "Idle Mean": "mean",
            " Down/Up Ratio": "mean",
            " Avg Fwd Segment Size": "mean",
            " Avg Bwd Segment Size": "mean",
            " SYN Flag Count": "sum",
            " ACK Flag Count": "sum",
            " RST Flag Count": "sum",
            " Protocol": "first",            # categorical → keep first (or mode)
            " Destination Port": "first",    # same
            " Label": lambda s: (s != "BENIGN").mean()  # attack ratio per window
        })

        ts = ts.fillna(0)  # fill gaps with 0 (no traffic)
        dfs.append(ts)"""
        
        break
  

    return dfs



def combining_dfs(dfs):  
    cols = dfs[0].columns
    # --- Step 4: Align DataFrames to only common columns ---
    aligned_dfs = [df.reindex(columns=sorted(cols)) for df in dfs]

    # --- Step 5: Concatenate into one big DataFrame ---
    big_df = pd.concat(aligned_dfs, ignore_index=True)


def main():
    # --- Settings ---
    csv_path = "datasets/03-11"
    preprocessing(csv_path)
    """load_dotenv()

    key = os.environ.get("SECRET_KEY")
    print(key)"""
    

if __name__ == "__main__":
    main()