
import pandas as pd
from sklearn.model_selection import train_test_split
import os, json

RAW_DATA_PATH = "data/tourism.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
CAT_COLS_PATH = "model/cat_cols.json"

df = pd.read_csv(RAW_DATA_PATH)
if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
with open(CAT_COLS_PATH, "w") as f: json.dump(cat_cols, f)

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_PATH, index=False)
pd.concat([X_test, y_test], axis=1).to_csv(TEST_PATH, index=False)

print("Data preprocessing completed!")
