import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = "Dataset/Online Retail.xlsx"
OUTPUT_DIR = "eda_outputs"


def load_data(path: str) -> pd.DataFrame:
    """Load and standardize the dataset for EDA."""
    df = pd.read_excel(path)
    df = df.copy()

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def display_overview(df: pd.DataFrame) -> None:
    print("\n=== Dataset Overview ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nSample records:")
    print(df.head().to_string(index=False))
    print("\nData types:")
    print(df.dtypes)


def display_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum().to_frame(name="Missing")
    missing["Missing_%"] = (missing["Missing"] / len(df)) * 100
    print("\n=== Missing Values ===")
    print(missing.sort_values("Missing", ascending=False).to_string())


def display_numeric_summary(df: pd.DataFrame) -> None:
    numeric_cols = ["Quantity", "UnitPrice", "TotalPrice"]
    print("\n=== Numeric Summary ===")
    print(df[numeric_cols].describe().round(2).to_string())


def display_country_summary(df: pd.DataFrame) -> None:
    country_summary = (
        df.groupby("Country")
        .agg(
            Transactions=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
            TotalSales=("TotalPrice", "sum"),
            AvgOrderValue=("TotalPrice", "mean"),
        )
        .sort_values(["Transactions", "Customers"], ascending=False)
    )

    print("\n=== Country-Level Summary ===")
    print(country_summary.head(10).round(2).to_string())


def display_product_summary(df: pd.DataFrame) -> None:
    top_products = (
        df.groupby("Description")
        .agg(UnitsSold=("Quantity", "sum"), Revenue=("TotalPrice", "sum"))
        .sort_values(["UnitsSold", "Revenue"], ascending=False)
        .head(10)
    )
    print("\n=== Top Products ===")
    print(top_products.round(2).to_string())


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.index, y=missing.values, palette="viridis")
    plt.title("Missing Values by Column")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "missing_values.png"), dpi=300)
    plt.close()


def plot_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["Quantity"], bins=40, kde=True, ax=axes[0])
    axes[0].set_title("Quantity Distribution")
    sns.histplot(df["UnitPrice"], bins=40, kde=True, ax=axes[1])
    axes[1].set_title("Unit Price Distribution")
    sns.histplot(df["TotalPrice"], bins=40, kde=True, ax=axes[2])
    axes[2].set_title("Total Price Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distributions.png"), dpi=300)
    plt.close()


def plot_country_sales(df: pd.DataFrame) -> None:
    country_sales = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=country_sales.values, y=country_sales.index, palette="magma")
    plt.title("Top 10 Countries by Total Sales")
    plt.xlabel("Total Sales")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "country_sales.png"), dpi=300)
    plt.close()


def plot_monthly_revenue(df: pd.DataFrame) -> None:
    monthly = (
        df.assign(month=df["InvoiceDate"].dt.to_period("M"))
        .groupby("month")["TotalPrice"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(12, 5))
    monthly.plot(kind="line", marker="o", linewidth=2)
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_revenue.png"), dpi=300)
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)

    display_overview(df)
    display_missing_values(df)
    display_numeric_summary(df)
    display_country_summary(df)
    display_product_summary(df)

    print("\n=== Data Quality Checks ===")
    print(f"Negative Quantity count: {(df['Quantity'] < 0).sum()}")
    print(f"Negative UnitPrice count: {(df['UnitPrice'] < 0).sum()}")
    print(f"Zero UnitPrice count: {(df['UnitPrice'] == 0).sum()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique invoices: {df['InvoiceNo'].nunique()}")

    plot_missing_values(df)
    plot_distribution(df)
    plot_country_sales(df)
    plot_monthly_revenue(df)

    print(f"\nEDA plots saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 120
    main()