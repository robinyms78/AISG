import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path: str) -> pd.DataFrame:
    # Load the Excel file
    df = pd.read_excel(path)

    # Display first 5 rows
    print(df.head())

    return df

def display_overview(df: pd.DataFrame) -> None:
    # Dataset information
    print(df.info())

    # Statistical summary
    print(df.describe())

    # Column names
    print(df.columns)

def check_missing_values(df: pd.DataFrame) -> None:
    missing_values = df.isnull().sum()
    print(missing_values)

    missing_percentage = df.isnull().sum() / len(df) * 100
    print(missing_percentage.sort_values(ascending=False))

    plt.figure(figsize=(10, 5))

    missing_values[missing_values > 0].plot(
        kind="bar"
    )

    plt.title("Missing Values by Column")
    plt.ylabel("Number of Missing Values")
    plt.show()

def check_duplicates(df: pd.DataFrame) -> None:
    print("Number of duplicate rows:", df.duplicated().sum())
    duplicates = df[df.duplicated()]
    print(duplicates.head())

def check_numerical_quantity(df: pd.DataFrame) -> None:
    # Display descriptive statistics
    print(df[["Quantity", "UnitPrice"]].describe())
    
    # Create 1 row and 2 columns of subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Quantity distribution
    sns.histplot(
        data=df,
        x="Quantity",
        bins=100,
        ax=axes[0]
    )
    
    axes[0].set_title("Distribution of Quantity")
    axes[0].set_xlim(-50, 200)
    axes[0].set_xlabel("Quantity")
    
    # Unit Price distribution
    sns.histplot(
        data=df,
        x="UnitPrice",
        bins=100,
        ax=axes[1]
    )
    
    axes[1].set_title("Distribution of Unit Price")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Unit Price")
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Display figure
    plt.show()

def check_negative_quantity(df: pd.DataFrame) -> None:
    print("Negative Quantity:",
      (df["Quantity"] < 0).sum())

    returns = df[df["Quantity"] < 0]
    print(returns.head())

    cancellations = df[
    df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    print("Number of cancellation records:",
        len(cancellations))

def create_total_sales_column(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    print(df.head())
    print(df.describe())

    return df

def analyse_sales_by_country(df: pd.DataFrame) -> None:
    country_sales = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
    )
    print(country_sales.head(10))

    plt.figure(figsize=(12, 6))
    country_sales.head(10).plot(
        kind="bar"
    )

    plt.title("Top 10 Countries by Total Sales")
    plt.xlabel("Country")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.show()

def analyze_best_selling_products(df: pd.DataFrame) -> None:
    top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    )
    print(top_products.head(10))

    plt.figure(figsize=(12, 6))
    top_products.head(10).sort_values().plot(
        kind="barh"
    )

    plt.title("Top 10 Products by Quantity Sold")
    plt.xlabel("Quantity Sold")
    plt.show()

def analyze_monthly_sales(df: pd.DataFrame) -> None:
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour

    monthly_sales = (
    df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"]
    .sum()
    )

    monthly_sales.index = monthly_sales.index.astype(str)

    plt.figure(figsize=(12, 6))
    plt.plot(
        monthly_sales.index,
        monthly_sales.values,
        marker="o"
    )
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def analyze_hourly_sales(df: pd.DataFrame) -> None:
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour

    hourly_sales = (
    df.groupby("Hour")["TotalPrice"]
    .sum()
    )

    plt.figure(figsize=(10, 5))
    hourly_sales.plot(
        kind="bar"
    )

    plt.title("Sales by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Total Sales")
    plt.show()

def analyze_customers_by_sales(df: pd.DataFrame) -> None:
    unique_customers = df["CustomerID"].nunique()
    print("Number of unique customers:", unique_customers)

    customer_sales = (
        df.groupby("CustomerID")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
    )
    print(customer_sales.head(10))

    plt.figure(figsize=(12, 6))
    customer_sales.head(10).plot(
        kind="bar"
    )

    plt.title("Top 10 Customers by Total Sales")
    plt.xlabel("Customer ID")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.show()

def analyze_customers_by_transaction_count(df: pd.DataFrame) -> None:
    customer_transaction_counts = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )
    print(customer_transaction_counts.head(10))

    plt.figure(figsize=(12, 6))
    customer_transaction_counts.head(10).plot(
        kind="bar"
    )

    plt.title("Top 10 Customers by Transaction Count")
    plt.xlabel("Customer ID")
    plt.ylabel("Transaction Count")
    plt.xticks(rotation=45)
    plt.show()


def main() -> None:
    data_path = "Dataset/Online Retail.xlsx"
    df = load_data(data_path)

    display_overview(df)
    check_missing_values(df)
    check_duplicates(df)
    check_numerical_quantity(df)
    check_negative_quantity(df) 
    df = create_total_sales_column(df)
    analyse_sales_by_country(df)
    analyze_best_selling_products(df)
    analyze_monthly_sales(df)
    analyze_hourly_sales(df)
    analyze_customers_by_sales(df)
    analyze_customers_by_transaction_count(df)


if __name__ == "__main__":
    main()