import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from the specified path."""
    df = pd.read_excel(path)
    return df

def process_data(df: pd.DataFrame) -> None:
    """Process the dataset by removing duplicates and handling missing values."""
    # Include data from UK
    df = df[df['Country'] == 'United Kingdom']

    # Handle missing CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)

    # Handle missing Description
    df.dropna(subset=['Description'], inplace=True)

    # Drop InvoiceNo that does not start with 'C' (cancellations)
    cancellations = df[
        df["InvoiceNo"].astype(str).str.startswith("C")
        ]
    df.drop(cancellations.index, inplace=True)

    # Drop quantity less than or equal to 0
    df = df[df["Quantity"] > 0]

    # Drop UnitPrice less than or equal to 0
    df = df[df["UnitPrice"] > 0]    

    # Save processed data to a csv file
    df.to_csv("Dataset/processed_data.csv", index=False)
    print(df.head())
    print(df.shape)

    return df

def main():
    # Example usage
    path = "Dataset/Online Retail.xlsx"
    df = load_data(path)
    df = process_data(df)


if __name__ == "__main__":
    main()