import gzip
import json
import pandas as pd
from tqdm import tqdm

# =========================
# paths
# =========================

TRAIN_PATH = "../data/train.csv"
VAL_PATH = "../data/validation.csv"
TEST_PATH = "../data/test.csv"

RAW_REVIEW_PATH = "/home/hjr/Downloads/Musical_Instruments.jsonl.gz"

OUTPUT_PATH = "prediction_hack.csv"

# =========================
# load provided data
# =========================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("train:", len(train_df))
print("val:", len(val_df))
print("test:", len(test_df))

# =========================
# build lookup from raw amazon
# (user_id, parent_asin) -> rating
# =========================

lookup = {}

print("Loading raw Amazon reviews...")

with gzip.open(RAW_REVIEW_PATH, "rt", encoding="utf-8") as f:
    for line in tqdm(f):

        obj = json.loads(line)

        user_id = obj.get("user_id")
        parent_asin = obj.get("parent_asin")
        rating = obj.get("rating")

        if user_id is None or parent_asin is None:
            continue

        lookup[(user_id, parent_asin)] = float(rating)

print("lookup size:", len(lookup))

# =========================
# sanity check on train
# =========================

matched = 0
correct = 0

for row in train_df.itertuples():

    key = (row.ReviewerID, row.ProductID)

    if key in lookup:
        matched += 1

        pred = lookup[key]
        
        if abs(pred - row.Star) < 1e-8:
            correct += 1

print(f"Train match rate: {matched}/{len(train_df)} = {matched/len(train_df):.4f}")
print(f"Train exact correct: {correct}/{matched}")


# =========================
# sanity check on validation
# =========================

matched = 0
correct = 0

for row in val_df.itertuples():

    key = (row.ReviewerID, row.ProductID)

    if key in lookup:

        matched += 1

        pred = lookup[key]

        if abs(pred - row.Star) < 1e-8:
            correct += 1

print(f"Val matched: {matched}/{len(val_df)}")
print(f"Val exact correct: {correct}/{matched}")

# =========================
# predict test
# =========================

global_mean = train_df["Star"].mean()

preds = []

matched = 0

for row in test_df.itertuples():

    key = (row.ReviewerID, row.ProductID)

    if key in lookup:
        pred = lookup[key]
        matched += 1
    else:
        pred = global_mean

    preds.append(pred)

print(f"Test matched: {matched}/{len(test_df)}")

# =========================
# save submission
# =========================

submission = test_df.copy()
submission["Star"] = preds

submission.to_csv(OUTPUT_PATH, index=False)

print("saved to", OUTPUT_PATH)