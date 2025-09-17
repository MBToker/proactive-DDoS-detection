from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# --- Settings ---
csv_path = "datasets/03-11"
load_dotenv()

key = os.environ.get("SECRET_KEY")
print(key)


# --- Step 1: Load all CSVs into list of DataFrames ---
all_files = [f for f in listdir(csv_path) if isfile(join(csv_path, f))]
dfs = []

pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.width", 200)          # wider console

for file in all_files:
    file_path = join(csv_path, file)
    print(f"[LOAD] {file_path}")
    temp_df = pd.read_csv(file_path)
    dfs.append(temp_df)

# --- Step 2: Collect column info ---
col_map = {file: set(df.columns) for file, df in zip(all_files, dfs)}

union_cols = set().union(*col_map.values())
common_cols = set.intersection(*col_map.values())

print("\n=== COLUMN REPORT ===")
print(f"Total files loaded: {len(all_files)}")
print(f"Union of all columns: {len(union_cols)}")
print(f"Common columns across all files: {len(common_cols)}\n")

# --- Step 3: Report per-file differences ---
for file, df in zip(all_files, dfs):
    extra = set(df.columns) - common_cols
    missing = common_cols - set(df.columns)
    if extra or missing:
        print(f"[⚠️] {file}")
        if extra:
            print(f"    Extra-only columns: {sorted(extra)}")
        if missing:
            print(f"    Missing columns: {sorted(missing)}")
    else:
        print(f"[✅] {file} has exactly the common set of columns")

# --- Step 4: Align DataFrames to only common columns ---
aligned_dfs = [df.reindex(columns=sorted(common_cols)) for df in dfs]

# --- Step 5: Concatenate into one big DataFrame ---
big_df = pd.concat(aligned_dfs, ignore_index=True)

labels = big_df[' Label'].unique()
print(labels)