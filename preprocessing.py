from os import listdir
from os.path import isfile, join
import os, re, tempfile
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pprint import pprint
from sqlalchemy import create_engine, text
from sqlalchemy.types import DateTime, Float, Integer
import re


# -----------------------------
# Unified feature set (for both)
# -----------------------------
UNIFIED_FEATURES = [
    # time
    "timestamp",

    # core volume/throughput
    "flow_duration", "total_fwd_packets", "total_backward_packets",
    "flow_bytes_s", "flow_packets_s",

    # packet sizes (means + spread/extremes)
    "average_packet_size", "packet_length_mean", "packet_length_std",
    "fwd_packet_length_mean", "bwd_packet_length_mean",
    "min_packet_length", "max_packet_length", "packet_length_variance",

    # timing / burstiness (means + spread/extremes)
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_mean", "fwd_iat_std", "bwd_iat_mean", "bwd_iat_std",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean",  "idle_std",  "idle_max",  "idle_min",

    # asymmetry
    "down_up_ratio", "avg_fwd_segment_size", "avg_bwd_segment_size",

    # flags (broader set)
    "syn_flag_count", "ack_flag_count", "rst_flag_count",
    "psh_flag_count", "urg_flag_count", "cwe_flag_count", "ece_flag_count",

    # bulk / subflow / header
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_avg_bulk_rate",
    "bwd_avg_bytes_bulk", "bwd_avg_packets_bulk", "bwd_avg_bulk_rate",
    "subflow_fwd_packets", "subflow_fwd_bytes",
    "subflow_bwd_packets", "subflow_bwd_bytes",
    "fwd_header_length", "bwd_header_length", "fwd_header_length_1",

    # other traffic indicators
    "init_win_bytes_forward", "init_win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",

    # categorical context
    "protocol", "destination_port",

    # label (optional)
    "label",
]


def selecting_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    print("----- 1. Selecting Final (Unified) Feature Set -----")
    # keep only columns that exist in df
    available = [c for c in UNIFIED_FEATURES if c in df.columns]
    # put timestamp first if present
    if "timestamp" in available:
        available = ["timestamp"] + [c for c in available if c != "timestamp"]
    print(f"Selected {len(available)} / {len(UNIFIED_FEATURES)} columns.")
    return df[available].copy()


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
    protected_cols=("protocol", "destination_port", "label"),
    return_report=True
):
    print("----- 2. Handling Missing Values In Data -----")
    print("----- Missing Report -----")
    df = df.copy()
    col_summary, row_summary = missing_summary(df)

    # numeric-like columns to smooth/interp
    interp_cols = [
        "flow_duration", "total_fwd_packets", "total_backward_packets",
        "flow_bytes_s", "flow_packets_s", "average_packet_size",
        "packet_length_mean", "packet_length_std", "packet_length_variance",
        "fwd_packet_length_mean", "bwd_packet_length_mean",
        "min_packet_length", "max_packet_length",
        "down_up_ratio", "avg_fwd_segment_size", "avg_bwd_segment_size",
        "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
        "fwd_iat_mean", "fwd_iat_std", "bwd_iat_mean", "bwd_iat_std",
        "active_mean", "active_std", "active_max", "active_min",
        "idle_mean", "idle_std", "idle_max", "idle_min",
        "fwd_header_length", "bwd_header_length", "fwd_header_length_1",
        "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_avg_bulk_rate",
        "bwd_avg_bytes_bulk", "bwd_avg_packets_bulk", "bwd_avg_bulk_rate",
        "subflow_fwd_packets", "subflow_fwd_bytes",
        "subflow_bwd_packets", "subflow_bwd_bytes",
        "init_win_bytes_forward", "init_win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "syn_flag_count", "ack_flag_count", "rst_flag_count",
        "psh_flag_count", "urg_flag_count", "cwe_flag_count", "ece_flag_count",
    ]

    # ffill-preferred columns (incl. categoricals / flags that can be sticky)
    ffill_cols = [
        "protocol", "destination_port"
    ]

    categoricals = ["protocol", "destination_port"]

    # filter existing
    interp_cols = [c for c in interp_cols if c in df.columns]
    ffill_cols  = [c for c in ffill_cols  if c in df.columns]
    categoricals = [c for c in categoricals if c in df.columns]

    # 1) missing fractions
    miss_frac = df.isna().mean().to_dict()

    # 2) drop heavily missing (except protected)
    drop_cols = [col for col, frac in miss_frac.items() if frac >= high_thresh and col not in protected_cols]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        interp_cols = [c for c in interp_cols if c in df.columns]
        ffill_cols  = [c for c in ffill_cols  if c in df.columns]
        categoricals = [c for c in categoricals if c in df.columns]

    # 3) numeric-ish: ffill → interpolate → median fallback
    if interp_cols:
        df[interp_cols] = df[interp_cols].ffill(limit=ffill_limit)
        df[interp_cols] = df[interp_cols].interpolate(method="linear", limit=ffill_limit, limit_direction="both")
        medians = df[interp_cols].median(numeric_only=True)
        df[interp_cols] = df[interp_cols].fillna(medians)

    # 4) sticky categoricals
    for col in ffill_cols:
        df[col] = df[col].ffill(limit=ffill_limit)
        if df[col].isna().any():
            try:
                mode_val = df[col].mode(dropna=True).iloc[0]
                df[col] = df[col].fillna(mode_val)
            except Exception:
                df[col] = df[col].fillna("Unknown" if col in categoricals else 0)

    # 5) label
    if "label" in df.columns:
        if mode == "train":
            df = df.dropna(subset=["label"])
        elif mode == "inference":
            df["label"] = df["label"].fillna("Unknown")
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


def _drop_fully_empty_bins(agg: pd.DataFrame) -> pd.DataFrame:
    # If a bin has zero counts and all mean-like features are NaN, drop it.
    count_like = [
        "total_fwd_packets","total_backward_packets","total_packets",
        "syn_flag_count","ack_flag_count","rst_flag_count",
        "psh_flag_count","urg_flag_count","cwe_flag_count","ece_flag_count",
        "subflow_fwd_packets","subflow_bwd_packets","act_data_pkt_fwd"
    ]
    mean_like = [
        "average_packet_size","packet_length_mean","packet_length_std","packet_length_variance",
        "fwd_packet_length_mean","bwd_packet_length_mean",
        "min_packet_length","max_packet_length",
        "flow_iat_mean","flow_iat_std","flow_iat_max","flow_iat_min",
        "fwd_iat_mean","fwd_iat_std","bwd_iat_mean","bwd_iat_std",
        "active_mean","active_std","active_max","active_min",
        "idle_mean","idle_std","idle_max","idle_min",
        "down_up_ratio","avg_fwd_segment_size","avg_bwd_segment_size",
        "flow_duration_s","flow_bytes_s","flow_packets_s",
        "fwd_header_length","bwd_header_length","fwd_header_length_1",
        "fwd_avg_bytes_bulk","fwd_avg_packets_bulk","fwd_avg_bulk_rate",
        "bwd_avg_bytes_bulk","bwd_avg_packets_bulk","bwd_avg_bulk_rate",
        "subflow_fwd_bytes","subflow_bwd_bytes",
        "init_win_bytes_forward","init_win_bytes_backward","min_seg_size_forward"
    ]

    present_counts = pd.Series(0, index=agg.index)
    for c in count_like:
        if c in agg.columns:
            present_counts = present_counts.add(agg[c].fillna(0), fill_value=0)

    if any(c in agg.columns for c in mean_like):
        all_means_nan = agg[[c for c in mean_like if c in agg.columns]].isna().all(axis=1)
    else:
        all_means_nan = pd.Series(False, index=agg.index)

    mask_empty = present_counts.eq(0) & all_means_nan
    return agg.loc[~mask_empty]


def resampling_by_protocol(df: pd.DataFrame, interval: str = "5s",
                           align_grid: bool = True,  # <- new
                           ffill_categoricals: bool = True,
                           drop_empty_bins: bool = False):
    df = df.copy()

    # --- timestamp -> index
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    if "protocol" not in df.columns:
        raise ValueError("protocol column is required for protocol-grouped resampling")

    # helpers
    if "flow_duration" in df.columns:
        df["flow_duration_s"] = pd.to_numeric(df["flow_duration"], errors="coerce").fillna(0) / 1e6
    else:
        df["flow_duration_s"] = 0

    if ("total_fwd_packets" in df.columns) or ("total_backward_packets" in df.columns):
        df["total_packets"] = pd.to_numeric(df.get("total_fwd_packets", 0), errors="coerce").fillna(0) + \
                              pd.to_numeric(df.get("total_backward_packets", 0), errors="coerce").fillna(0)

    sum_cols = [
        "total_fwd_packets","total_backward_packets","total_packets",
        "syn_flag_count","ack_flag_count","rst_flag_count",
        "psh_flag_count","urg_flag_count","cwe_flag_count","ece_flag_count",
        "subflow_fwd_packets","subflow_bwd_packets","subflow_fwd_bytes","subflow_bwd_bytes",
        "act_data_pkt_fwd"
    ]
    mean_cols = [
        "average_packet_size","packet_length_mean","packet_length_std","packet_length_variance",
        "fwd_packet_length_mean","bwd_packet_length_mean",
        "fwd_iat_mean","fwd_iat_std","bwd_iat_mean","bwd_iat_std",
        "flow_iat_mean","flow_iat_std",
        "active_mean","active_std","idle_mean","idle_std",
        "down_up_ratio","avg_fwd_segment_size","avg_bwd_segment_size",
        "fwd_header_length","bwd_header_length","fwd_header_length_1",
        "fwd_avg_bytes_bulk","fwd_avg_packets_bulk","fwd_avg_bulk_rate",
        "bwd_avg_bytes_bulk","bwd_avg_packets_bulk","bwd_avg_bulk_rate",
        "init_win_bytes_forward","init_win_bytes_backward","min_seg_size_forward", "flow_bytes_s",
    ]
    max_cols = ["flow_iat_max","active_max","idle_max","max_packet_length"]
    min_cols = ["flow_iat_min","active_min","idle_min","min_packet_length"]

    agg_map = {}
    for c in sum_cols:
        if c in df.columns: agg_map[c] = "sum"
    for c in mean_cols:
        if c in df.columns: agg_map[c] = "mean"
    for c in max_cols:
        if c in df.columns: agg_map[c] = "max"
    for c in min_cols:
        if c in df.columns: agg_map[c] = "min"

    if "destination_port" in df.columns:
        agg_map["destination_port"] = lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan
    if "label" in df.columns:
        agg_map["label"] = lambda s: (
            "BENIGN" if s.dropna().eq("BENIGN").all()
            else (s[s.ne("BENIGN")].mode().iloc[0] if not s[s.ne("BENIGN")].mode().empty else "BENIGN")
        )

    agg_map["flow_duration_s"] = "sum"

    # --- per-protocol resample (aligned bin edges via origin="epoch")
    grp = (df.groupby("protocol", dropna=False)
             .resample(interval, label="right", closed="right", origin="epoch")
             .agg(agg_map))

    # --- global grid alignment (same timestamps for every protocol)
    if align_grid:
        protos = grp.index.get_level_values("protocol").unique()
        # use the original df time span for the grid
        start = df.index.min().floor(interval)
        end   = df.index.max().ceil(interval)
        grid = pd.date_range(start, end, freq=interval)
        full_idx = pd.MultiIndex.from_product([protos, grid], names=["protocol","timestamp"])
        grp = grp.reindex(full_idx)

    # derive flow_packets_s AFTER reindexing/filling
    if "total_packets" in grp.columns:
        dur = grp["flow_duration_s"].where(grp["flow_duration_s"] > 0)
        grp["flow_packets_s"] = (grp["total_packets"] / dur).where(dur.notna())

    grp = grp.replace([np.inf, -np.inf], np.nan)

    # fill count-like with 0 (no flows in bin ⇒ 0)
    count_like = [c for c in [
        "total_fwd_packets","total_backward_packets","total_packets",
        "syn_flag_count","ack_flag_count","rst_flag_count",
        "psh_flag_count","urg_flag_count","cwe_flag_count","ece_flag_count",
        "subflow_fwd_packets","subflow_bwd_packets","act_data_pkt_fwd",
        "flow_duration_s"
    ] if c in grp.columns]
    for c in count_like:
        grp[c] = grp[c].fillna(0)

    # optional: forward-fill categorical context within protocol
    if ffill_categoricals and "destination_port" in grp.columns:
        grp["destination_port"] = grp.groupby(level=0)["destination_port"].ffill(limit=1)

    # optionally drop bins that remain completely empty
    if drop_empty_bins:
        grp = _drop_fully_empty_bins(grp)

    grp = grp.reset_index()  # protocol, timestamp, ...
    num_cols = grp.select_dtypes(include=[np.number]).columns
    grp[num_cols] = grp[num_cols].round(2)
    return grp



def make_engine_from_env():
    load_dotenv()  # reads .env in current working dir
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    pwd  = os.getenv("PGPASSWORD", "")
    db   = os.getenv("PGDATABASE", "postgres")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def ensure_schema(engine, schema: str):
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def save_fast_with_copy(engine, df: pd.DataFrame, schema: str, table: str,
                        if_exists: str = "replace", make_unlogged: bool = True,
                        sanitize_cols: bool = True):
    """Create table from df schema, then bulk-load rows using COPY (very fast)."""
    # (optional) sanitize weird column names for DB friendliness
    if sanitize_cols:
        safe_cols = []
        for c in df.columns:
            cc = re.sub(r"\W+", "_", c.strip().lower()).strip("_").lower()  # letters/digits/_ only
            safe_cols.append(cc or "col")
        df = df.copy()
        df.columns = safe_cols

    with engine.begin() as conn:
        if if_exists not in {"replace", "append", "fail"}:
            raise ValueError("if_exists must be 'replace' | 'append' | 'fail'")
        if if_exists == "replace":
            conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table}"'))

        # Create an empty table from the DataFrame schema
        df.head(0).to_sql(name=table, con=conn, schema=schema, if_exists="append", index=False)

        # Optionally flip to UNLOGGED for raw/preprocessed staging (faster writes)
        if make_unlogged:
            conn.execute(text(f'ALTER TABLE "{schema}"."{table}" SET UNLOGGED'))

    # Dump to a temp CSV and COPY it (fastest path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8", newline="") as tmp:
        csv_path = tmp.name
        df.to_csv(tmp, index=False)

    try:
        raw = engine.raw_connection()  # psycopg2 connection
        try:
            with raw.cursor() as cur, open(csv_path, "r", encoding="utf-8") as f:
                cur.copy_expert(
                    f'COPY "{schema}"."{table}" FROM STDIN WITH (FORMAT CSV, HEADER TRUE)',
                    f
                )
            raw.commit()
        finally:
            raw.close()
    finally:
        try: os.remove(csv_path)
        except OSError: pass


def protocol_overview(df: pd.DataFrame):
    g = df.groupby("protocol", dropna=False)
    view = g.agg(
        bins=("timestamp","count"),
        start=("timestamp","min"),
        end=("timestamp","max"),
        pct_nonzero_pkts=("total_packets", lambda s: float((s>0).mean()*100) if "total_packets" in df.columns else np.nan),
        pct_missing_any=("timestamp", lambda _: float(g.apply(lambda x: x.drop(columns=["timestamp"]).isna().any(axis=1).mean()*100).mean()))
    ).sort_values("bins", ascending=False)
    print(view)
    return view


def preprocessing(csv_path, interval="5s"):
    raw_schema = "raw"
    preprocessed_schema = "preprocessed"
    engine = make_engine_from_env()
    ensure_schema(engine, raw_schema)
    ensure_schema(engine, preprocessed_schema)

    all_files = [f for f in listdir(csv_path) if isfile(join(csv_path, f))]
    perfile_rs = []

    for file in all_files:
        table_name = file.replace(".csv", "").lower()
        file_path = join(csv_path, file)
        print(f"[LOAD] {file_path}")

        temp_df = pd.read_csv(file_path)

        # sanitize columns early
        temp_df.columns = [
            re.sub(r"\W+", "_", c.strip().lower()).strip("_")
            for c in temp_df.columns
        ]

        # select unified features that exist
        temp_df = selecting_feature_set(temp_df)

        # handle missings
        temp_df, report = handling_missing_vals(temp_df)
        pprint(report, sort_dicts=False, width=100)

        # resample per file
        rs = resampling_by_protocol(temp_df, interval)
        rs["source_file"] = table_name  # lineage (optional)
        perfile_rs.append(rs)

    # combine already-resampled files (no second global resample)
    big_df = pd.concat(perfile_rs, ignore_index=True).sort_values("timestamp")

    print(f"[SAVE] preprocessed -> {preprocessed_schema}.combined (COPY)")
    save_fast_with_copy(
        engine,
        df=big_df,
        schema=preprocessed_schema,
        table="combined",
        if_exists="replace",
        make_unlogged=True,
        sanitize_cols=True
    )

    # diagnostics
    total_bins = len(big_df)
    empty_bins = big_df.drop(columns=["timestamp"]).isna().all(axis=1).sum()
    print(f"Empty bins (all-NaN rows): {empty_bins}/{total_bins} ({empty_bins/max(total_bins,1):.1%})")
    print("Top sparsity after per-file resample then combine:")
    print((big_df.isna().mean().sort_values(ascending=False)*100).round(2).head(10))
    
    # after big_df = ...
    print("\n=== Protocol overview ===")
    protocol_overview(big_df)

    return big_df


def main():
    # --- Settings ---
    csv_path = "datasets/03-11"
    preprocessing(csv_path)


if __name__ == "__main__":
    main()
