from os import listdir
from os.path import isfile, join
import os, re, tempfile
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pprint import pprint
from sqlalchemy import create_engine, text
from sqlalchemy.types import DateTime, Float, Integer


def selecting_feature_set(df):
    print("----- 1. Selecting Final Feature Set -----")
    selected_features = [
        "timestamp",

        # Traffic volume & throughput
        "flow_duration",
        "total_fwd_packets",
        "total_backward_packets",
        "flow_bytes/s",
        "flow_packets/s",
        "average_packet_size",

        # Packet size stats
        "packet_length_mean",
        "packet_length_std",
        "fwd_packet_length_mean",
        "bwd_packet_length_mean",

        # Timing / burstiness 
        "flow_iat_mean",
        "flow_iat_std",
        "fwd_iat_mean",
        "bwd_iat_mean",
        "active_mean",
        "idle_mean",

        # Asymmetry 
        "down/up_ratio",
        "avg_fwd_segment_size",
        "avg_bwd_segment_size",

        # TCP flag behavior 
        "syn_flag_count",
        "ack_flag_count",
        "rst_flag_count",

        # Protocol context 
        "protocol",
        "destination_port",

        # Label for anomaly detection
        "label"
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
        "flow_duration", "total_fwd_packets", "total_backward_packets",
        "flow_bytes/s", "flow_packets/s", "average_packet_size",
        "packet_length_mean", "packet_length_std",
        "fwd_packet_length_mean", "bwd_packet_length_mean",
        "down/up_ratio", "avg_fwd_segment_size", "avg_bwd_segment_size"
    ]

    ffill_cols = [
        "flow_iat_mean", "flow_iat_std", "fwd_iat_mean", "bwd_iat_mean",
        "active_mean", "idle_mean",
        "syn_flag_count", "ack_flag_count", "rst_flag_count",
        "protocol", "destination_port"
    ]

    categoricals = ["protocol", "destination_port"]


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


def _mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan


def _top_attack_type(series: pd.Series):
    s = series.dropna().astype(str)
    att = s[s != "BENIGN"]
    if att.empty:
        return "BENIGN"
    m = att.mode()
    return m.iloc[0] if not m.empty else "BENIGN"


def resampling(df: pd.DataFrame, interval: str = "5s") -> pd.DataFrame:
    df = df.copy()

    # --- timestamp -> index (sabit grid & sağ kapalı pencere)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # --- birim düzeltmesi ve yardımcı kolonlar
    # CICDDoS2019: Flow Duration mikro-saniye -> saniye
    df["flow_duration_s"] = pd.to_numeric(df.get("flow_duration", 0), errors="coerce").fillna(0) / 1e6
    df["total_packets"] = pd.to_numeric(df.get("total_fwd_packets", 0), errors="coerce").fillna(0) + \
                          pd.to_numeric(df.get("total_backward_packets", 0), errors="coerce").fillna(0)

    # --- agregasyon
    agg = df.resample(interval, label="right", closed="right", origin="epoch").agg({
        "total_fwd_packets": "sum",
        "total_backward_packets": "sum",
        "total_packets": "sum",
        "syn_flag_count": "sum",
        "ack_flag_count": "sum",
        "rst_flag_count": "sum",
        "flow_duration_s": "sum",
        "average_packet_size": "mean",
        "packet_length_mean": "mean",
        "packet_length_std": "mean",
        "fwd_packet_length_mean": "mean",
        "bwd_packet_length_mean": "mean",
        "flow_iat_mean": "mean",
        "flow_iat_std": "mean",
        "fwd_iat_mean": "mean",
        "bwd_iat_mean": "mean",
        "active_mean": "mean",
        "idle_mean": "mean",
        "down/up_ratio": "mean",
        "avg_fwd_segment_size": "mean",
        "avg_bwd_segment_size": "mean",
        "protocol": lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan,
        "destination_port": lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan,
        "label": lambda s: (
            "BENIGN" if s.dropna().eq("BENIGN").all()
            else (s[s.ne("BENIGN")].mode().iloc[0] if not s[s.ne("BENIGN")].mode().empty else "BENIGN")
        ),
    })

    # --- oran/ hızlar (0 süreye karşı güvenli)
    dur = agg["flow_duration_s"].replace(0, np.nan)  # 0 saniyeyi NaN yap → bölme inf olmaz
    agg["flow_packets_s"] = agg["total_packets"] / dur
    # Gerçek byte toplamları yoksa bytes/s hesaplamasını atla; varsa burada kullan:
    # agg["flow_bytes_s"] = agg["total_bytes"] / dur

    # --- isim iyileştirme: label -> top_attack_type
    agg = agg.rename(columns={"label": "top_attack_type"})

    # --- inf/-inf -> NaN; sayısallar için makul doldurma
    agg = agg.replace([np.inf, -np.inf], np.nan)
    count_cols = ["total_fwd_packets","total_backward_packets","total_packets",
                  "syn_flag_count","ack_flag_count","rst_flag_count","flow_duration_s"]
    for c in count_cols:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0)

    # top_attack_type boşsa BENIGN
    if "top_attack_type" in agg.columns:
        agg["top_attack_type"] = agg["top_attack_type"].fillna("BENIGN")

    # --- timestamp'i kolona geri koy
    agg.index.name = "timestamp"
    agg = agg.reset_index()

    return agg


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
            cc = re.sub(r"\W+", "_", c).strip("_").lower()  # letters/digits/_ only
            safe_cols.append(cc or "col")
        df = df.copy()
        df.columns = safe_cols

    with engine.begin() as conn:
        if if_exists not in {"replace", "append", "fail"}:
            raise ValueError("if_exists must be 'replace' | 'append' | 'fail'")
        if if_exists == "replace":
            conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table}"'))

        # Create an empty table from the DataFrame schema
        # (0 rows => just DDL; much faster than inserting)
        df.head(0).to_sql(name=table, con=conn, schema=schema, if_exists="append", index=False)

        # Optionally flip to UNLOGGED for raw/preprocessed staging (faster writes)
        if make_unlogged:
            conn.execute(text(f'ALTER TABLE "{schema}"."{table}" SET UNLOGGED'))

    # Dump to a temp CSV and COPY it (fastest path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8", newline="") as tmp:
        csv_path = tmp.name
        # choose na_rep="" if you prefer blanks over "NaN"
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
      
  
def preprocessing(csv_path, interval="5s"):
    # DB settings
    raw_schema = "raw"
    preprocessed_schema = "preprocessed"
    engine = make_engine_from_env()
    ensure_schema(engine, raw_schema)
    ensure_schema(engine, preprocessed_schema)

    all_files = [f for f in listdir(csv_path) if isfile(join(csv_path, f))]
    dfs = []

    for file in all_files:
        table_name = file.replace(".csv", "").lower()
        file_path = join(csv_path, file)
        print(f"[LOAD] {file_path}")

        temp_df = pd.read_csv(file_path)
        temp_df.columns = ["_".join(c.strip().lower().split()) for c in temp_df.columns]
        temp_df = selecting_feature_set(temp_df)
        temp_df, report = handling_missing_vals(temp_df)
        pprint(report, sort_dicts=False, width=100)

        print(f"[SAVE] raw -> {raw_schema}.{table_name} (COPY)")
        save_fast_with_copy(
            engine,
            df=temp_df,
            schema=raw_schema,
            table=table_name,
            if_exists="replace",
            make_unlogged=True,     # speed boost for staging tables
            sanitize_cols=True      # replace odd chars like "/" with "_"
        )

        temp_df = resampling(temp_df, interval)
        
        print(f"[SAVE] preprocessed -> {preprocessed_schema}.{table_name} (COPY)")
        save_fast_with_copy(
            engine,
            df=temp_df,
            schema=preprocessed_schema,
            table=table_name,
            if_exists="replace",
            make_unlogged=True,     # speed boost for staging tables
            sanitize_cols=True      # replace odd chars like "/" with "_"
        )
        
        dfs.append(temp_df)
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