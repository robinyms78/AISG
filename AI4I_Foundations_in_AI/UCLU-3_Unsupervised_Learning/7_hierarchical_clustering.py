import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os
import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def prepare_data(path: str):
    # Load data
    df = pd.read_csv(path)
    print(df.head())
    print(df.info())
    print(df.describe())\

    # Transform the RFM values using logarithmic transformation to reduce skewness
    rfm_log = df[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1)
    print(rfm_log.head())

    # Normalize the transformed RFM values
    df_scaled = pd.DataFrame(normalize(rfm_log), columns=['Recency','Frequency','Monetary'])
    print(df_scaled.head())

    return df_scaled

def plot_distributions(df_scaled):
    # Plot and save the distributions of the scaled RFM values
    output_dir = 'eda_outputs/hierarchy_clustering'
    os.makedirs(output_dir, exist_ok=True)
    axes = df_scaled.hist(bins=20, figsize=(12, 4), edgecolor='black')
    for axis, column in zip(axes[0], df_scaled.columns):
        axis.set_title(f'{column} distribution')
        axis.set_xlabel(column)
        axis.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaled_rfm_histograms.png', bbox_inches='tight', dpi=150)
    plt.close()

def plot_dendrogram(df_scaled):
    output_dir = 'eda_outputs/hierarchy_clustering'
    os.makedirs(output_dir, exist_ok=True)
    linkage_matrix = hc.linkage(df_scaled, method='average')

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    hc.dendrogram(linkage_matrix)
    plt.xlabel("Customer index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hierarchy_clustering_average.png', bbox_inches='tight', dpi=150)
    plt.close()

def plot_3d_clusters(df_scaled):
    output_dir = 'eda_outputs/hierarchy_clustering'
    os.makedirs(output_dir, exist_ok=True)
    model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
    cluster_labels = model.fit_predict(df_scaled)

    # Plot the 3D scatter plot of the clusters
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.scatter(df_scaled.Recency, df_scaled.Frequency, df_scaled.Monetary, c=cluster_labels, cmap='viridis', linewidth=0.5)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    plt.savefig(f'{output_dir}/3d_cluster.png', bbox_inches='tight', dpi=150)
    plt.close()

    return cluster_labels

def plot_pca(df_scaled, cluster_labels):
    output_dir = 'eda_outputs/hierarchy_clustering'
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2)
    df_pca2 = pca.fit_transform(df_scaled)

    # Plot the PCA result
    (x, y) = (df_pca2[:,0], df_pca2[:,1])
    plt.figure(figsize=(12,12))
    plt.scatter(x, y, c=cluster_labels, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(f'{output_dir}/pca_rfm.png', bbox_inches='tight', dpi=150)
    plt.close()

def add_cluster_labels(df_scaled, cluster_labels):
    df_scaled['Cluster'] = cluster_labels
    print(df_scaled.head())


if __name__ == "__main__":
    path = "eda_outputs/large_customers.csv"
    df = prepare_data(path)
    plot_distributions(df)
    plot_dendrogram(df)
    cluster_labels = plot_3d_clusters(df)
    plot_pca(df, cluster_labels)
    add_cluster_labels(df, cluster_labels)