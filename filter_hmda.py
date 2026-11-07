import pandas as pd
import re

usecols = [
    "income", "loan_amount", "loan_to_value_ratio", "debt_to_income_ratio",
    "derived_race", "derived_ethnicity", "derived_sex",
    "action_taken", "loan_type", "loan_purpose", "lien_status", "occupancy_type",
]

def midpoint(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.replace('.', '', 1).isdigit():
        return float(s)
    nums = re.findall(r"\d+\.?\d*", s)
    if len(nums) == 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return None

chunks = []
target_n = 600_000
reader = pd.read_csv(
    "loan_types_1_year_2022.csv",
    usecols=usecols,
    chunksize=500_000,
    dtype=str,
    low_memory=False,
)

for chunk in reader:
    chunk["loan_type"] = pd.to_numeric(chunk["loan_type"], errors="coerce")
    chunk["loan_purpose"] = pd.to_numeric(chunk["loan_purpose"], errors="coerce")
    chunk["lien_status"] = pd.to_numeric(chunk["lien_status"], errors="coerce")
    chunk["occupancy_type"] = pd.to_numeric(chunk["occupancy_type"], errors="coerce")
    chunk["action_taken"] = pd.to_numeric(chunk["action_taken"], errors="coerce")

    c = chunk[
        (chunk["loan_type"] == 1) &
        (chunk["loan_purpose"] == 1) &
        (chunk["lien_status"] == 1) &
        (chunk["occupancy_type"] == 1) &
        (chunk["action_taken"].isin([1, 3]))
    ].copy()
    chunks.append(c)
    print(f"...processed a chunk, kept {len(c)} rows, running total {sum(len(x) for x in chunks)}")
    if sum(len(x) for x in chunks) >= target_n:
        break

df = pd.concat(chunks, ignore_index=True)
df["debt_to_income_ratio"] = df["debt_to_income_ratio"].apply(midpoint)
df["loan_to_value_ratio"] = pd.to_numeric(df["loan_to_value_ratio"], errors="coerce")
df["income"] = pd.to_numeric(df["income"], errors="coerce")
df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
df = df.dropna(subset=["income", "loan_amount", "debt_to_income_ratio", "loan_to_value_ratio"])
df["action_taken_binary"] = (df["action_taken"] == 1).astype(int)

df = df.sample(n=min(500_000, len(df)), random_state=11)
df.to_csv("hmda_sample.csv", index=False)
print(f"Wrote hmda_sample.csv: {len(df)} rows")
print(df["action_taken_binary"].value_counts())
print(df.groupby("derived_sex")["action_taken_binary"].mean())
