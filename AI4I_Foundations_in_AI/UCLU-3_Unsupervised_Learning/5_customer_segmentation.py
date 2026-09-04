import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_data(path: str) -> pd.DataFrame:
    # Load the CSV file
    df = pd.read_csv(path)
    
    # Display first 5 rows
    print(df.head())
    print(df.dtypes)

    return df

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Set CustomerID and StockCode as a categorical variable
    df['CustomerID'] = pd.Categorical(df['CustomerID'].astype(int))
    df['StockCode'] = pd.Categorical(df['StockCode'])

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Drop country column
    df = df.drop(columns=["Country"])

    # Extracting only the day information from the InvoiceDate column
    df['InvoiceDay'] = df.InvoiceDate.apply(lambda x: dt.datetime(x.year, x.month, x.day))

    # Define reference date for recency calculation
    ref_date = max(df.InvoiceDay) + dt.timedelta(1)

    # Calculate the total value for each transaction
    df['TotalSum'] = df.Quantity * df.UnitPrice

    # Calculate RFM
    rfm = df.groupby('CustomerID').agg({
     'InvoiceNo':'count',
     'TotalSum':'sum',
     'InvoiceDay': lambda x: ref_date - x.max()})

    rfm.rename(columns={
     'InvoiceNo':'Frequency',
     'TotalSum':'Monetary',
     'InvoiceDay':'Recency'
    }, inplace=True)

    rfm['Recency'] = rfm['Recency'].dt.days
    print(rfm.dtypes)

    return rfm

def transform_data(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1)
    
    return rfm_log

def visualize_plot(rfm: pd.DataFrame) -> None:
    # display the histogram for the dataset
    plt.figure(figsize=(10, 6))
    sns.histplot(data=rfm, x="Frequency", bins=10)
    plt.title("Distribution of Frequency")
    plt.xlabel("Frequency")  
    plt.savefig("eda_outputs/kmeans_clustering/frequency_distribution.png", bbox_inches="tight", dpi=150)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=rfm, x="Monetary", bins=10)
    plt.title("Distribution of Monetary")
    plt.xlabel("Monetary")  
    plt.savefig("eda_outputs/kmeans_clustering/monetary_distribution.png", bbox_inches="tight", dpi=150)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=rfm, x="Recency", bins=10)
    plt.title("Distribution of Recency")
    plt.xlabel("Recency")  
    plt.savefig("eda_outputs/kmeans_clustering/recency_distribution.png", bbox_inches="tight", dpi=150)

def find_optimal_clusters(rfm_log: pd.DataFrame) -> None:
    inertia = {}
    for k in range(1,11):
        mod = KMeans(n_clusters=k)
        mod.fit(rfm_log)
        inertia[k] = mod.inertia_
    
    # Create an elbow plot to determine the best k
    plt.figure(figsize=(10, 6))
    sns.pointplot(x=list(inertia.keys()), y=list(inertia.values()))
    plt.xlabel('K Numbers')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig('eda_outputs/kmeans_clustering/elbow_plot.png', bbox_inches='tight', dpi=150)

def train_kmeans(rfm_log: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters)
    model = kmeans.fit(rfm_log)
    rfm_log['Cluster'] = kmeans.labels_
    print(rfm_log.head())
    
    return model, rfm_log

def visualize_3d_clusters(rfm_log: pd.DataFrame) -> None:
    fig=plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.scatter(rfm_log.Recency, rfm_log.Frequency, 
                            rfm_log.Monetary, c=rfm_log.Cluster,
                            cmap='viridis', linewidth=0.5);
    plt.savefig('eda_outputs/kmeans_clustering/cluster_plot.png', bbox_inches='tight', dpi=150)

def visualize_2d_clusters(rfm_log: pd.DataFrame) -> None:
    features = ['Recency', 'Frequency', 'Monetary']
    pca = PCA(n_components=2)
    rfm_pca2 = pca.fit_transform(rfm_log[features])
    pca_df = pd.DataFrame(rfm_pca2, columns=['PC1', 'PC2'], index=rfm_log.index)
    pca_df['Cluster'] = rfm_log['Cluster'].to_numpy()
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA shape: {pca_df[['PC1', 'PC2']].shape}")
    print(f"Explained variance: {explained_variance:.2%}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
    plt.title('2D PCA Cluster Visualization')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend(title='Cluster')
    plt.savefig('eda_outputs/kmeans_clustering/2d_cluster_plot.png', bbox_inches='tight', dpi=150)
    plt.close()

def split_data(rfm_log: pd.DataFrame, model) -> None:
    features = ['Recency', 'Frequency', 'Monetary']
    cluster_centroids = pd.DataFrame(
        np.exp(model.cluster_centers_),
        columns=features
    )
    cluster_centroids['Cluster'] = cluster_centroids.index
    cluster_centroids = cluster_centroids.sort_values('Monetary')
    cluster_labels = {
        cluster_centroids.iloc[0]['Cluster']: 'small',
        cluster_centroids.iloc[1]['Cluster']: 'medium',
        cluster_centroids.iloc[2]['Cluster']: 'large'
    }
    print(cluster_centroids[features].round(3))

    rfm_original = rfm_log[features].apply(np.exp)
    rfm_original['Cluster'] = rfm_log['Cluster']
    rfm_original['CustomerType'] = rfm_original['Cluster'].map(cluster_labels)
    rfm_original['Recency'] = rfm_original['Recency'].round().astype(int)
    rfm_original['Frequency'] = rfm_original['Frequency'].round().astype(int)
    rfm_original['Monetary'] = rfm_original['Monetary'].round(2)

    os.makedirs('eda_outputs', exist_ok=True)
    for customer_type in ('small', 'large'):
        customers = rfm_original[rfm_original['CustomerType'] == customer_type]
        customers.to_csv(
            f'eda_outputs/{customer_type}_customers.csv',
            index_label='CustomerID'
        )
        print(f'Saved {len(customers)} {customer_type} customers')

def main(path: str):
    # Example usage
    df = load_data(path)
    df = prepare_data(df)
    df = transform_data(df)
    visualize_plot(df)
    find_optimal_clusters(df)
    model, df = train_kmeans(df, n_clusters=3)
    visualize_3d_clusters(df)
    visualize_2d_clusters(df)
    split_data(df, model)


if __name__ == "__main__":
    # Example usage
    data_path = "Dataset/processed_data.csv"  
    main(data_path)