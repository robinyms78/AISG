from pathlib import Path
import pandas as pd

# Import train and test data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Check shape
print(train_df.shape)
print(test_df.shape)

# Check number of numerical and categorical data
num_features = train_df.select_dtypes(include="number").shape[1]
cat_features = train_df.select_dtypes(include="object").shape[1]

print("Numerical features:", num_features)
print("Categorical features:", cat_features)