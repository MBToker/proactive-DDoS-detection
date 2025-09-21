from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import os
import numpy as np


def make_engine_from_env():
    load_dotenv()  # reads .env in current working dir
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    user = os.getenv("PGUSER", "postgres")
    pwd  = os.getenv("PGPASSWORD", "")
    db   = os.getenv("PGDATABASE", "postgres")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def get_table(schema_name, table_name):
    engine = make_engine_from_env()
    df = pd.read_sql(f"SELECT * FROM {schema_name}.{table_name}", engine)
    return df


def create_lags(df, lags, excluded_cols, time_col="timestamp"):
    df_lagged = df.copy()

    if time_col in df_lagged.columns:
        df_lagged[time_col] = pd.to_datetime(df_lagged[time_col], errors="coerce")
        df_lagged = df_lagged.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    included_cols = [c for c in df_lagged.columns if c not in excluded_cols]

    # Collect lagged columns in a dict
    lagged_data = {}
    for col in included_cols:
        for lag in lags:
            lagged_data[f"{col}_lag{lag}"] = df_lagged[col].shift(lag)

    lags_df = pd.DataFrame(lagged_data, index=df_lagged.index)
    df_lagged = pd.concat([df_lagged, lags_df], axis=1)

    k = int(max(lags)) if lags else 0
    if k > 0:
        df_lagged = df_lagged.iloc[k:].reset_index(drop=True)
    else:
        df_lagged = df_lagged.reset_index(drop=True)

    return df_lagged


def correlation_pipeline(df, excluded_cols ,threshold=0.4):
    targets = [target for target in df.columns if "lag" not in target and target not in excluded_cols]
    features_per_target = {}
    
    for target in targets:
        feats = [c for c in df.columns if c != target and "_lag" in c]
        corr = df[feats + [target]].corr(method="spearman")[target].drop(target)
        corr = corr.abs()
        corr = corr[abs(corr) >= threshold].sort_values(ascending=False)
        corr_keys = corr.index.tolist()
        features_per_target = {target: corr_keys}
        
    return features_per_target

        
def main():
    schema_name = "preprocessed"
    table_name = "combined"
    lags = [1, 2, 3, 6, 12]
    excluded_cols = ['timestamp', 'label', 'protocol', 'destination_port', 'source_file']
    
    df = get_table(schema_name, table_name)
    unique_protocols = df["protocol"].unique().tolist()
    
    for protocol_val in unique_protocols:
        grouped_df = df[df["protocol"] == protocol_val]
        lagged_df = create_lags(grouped_df, lags, excluded_cols)
        features_dict = correlation_pipeline(lagged_df, excluded_cols, threshold=0.6)
        print(features_dict)
        break

    


if __name__ == "__main__":
    main()
