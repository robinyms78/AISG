from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# Load data
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"

train_df = pd.read_csv(DATA_DIR / "train.csv")

# --------------------------------------------------
# Function to calculate outliers using IQR
# --------------------------------------------------
def count_outliers(series):
    # Only calculate outliers for numerical data
    if not pd.api.types.is_numeric_dtype(series):
        return np.nan

    # Remove missing values
    series = series.dropna()

    if len(series) == 0:
        return np.nan

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return len(outliers)


# --------------------------------------------------
# Create feature analysis table
# --------------------------------------------------
analysis = []

for column in train_df.columns:

    total = len(train_df)

    # Missing values
    missing_count = train_df[column].isna().sum()
    missing_pct = (missing_count / total) * 100

    # Unique values
    unique_count = train_df[column].nunique()

    # Outliers
    outlier_count = count_outliers(train_df[column])

    if pd.isna(outlier_count):
        outlier_pct = np.nan
    else:
        non_missing_count = train_df[column].notna().sum()

        if non_missing_count > 0:
            outlier_pct = (
                outlier_count /
                non_missing_count
            ) * 100
        else:
            outlier_pct = np.nan

    analysis.append({
        "Feature": column,
        "Data Type": train_df[column].dtype,
        "Missing Count": missing_count,
        "Missing %": missing_pct,
        "Unique Values": unique_count,
        "Outlier Count": outlier_count,
        "Outlier %": outlier_pct
    })


analysis_df = pd.DataFrame(analysis)

# --------------------------------------------------
# Sort by missing percentage
# --------------------------------------------------
analysis_df = analysis_df.sort_values(
    by="Missing %",
    ascending=False
)

# Display
print(analysis_df.to_string(index=False))