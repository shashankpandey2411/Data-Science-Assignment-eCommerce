import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
customers_df = pd.read_csv('Customers.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Feature Engineering
def prepare_features(customers_df, transactions_df):
    # Convert dates to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Calculate customer metrics
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',  # Number of transactions
        'TotalValue': ['sum', 'mean'],  # Total spend and average spend
        'Quantity': ['sum', 'mean']  # Total quantity and average quantity
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['CustomerID', 
                              'transaction_count', 
                              'total_spend', 
                              'avg_transaction_value',
                              'total_quantity',
                              'avg_quantity']
    
    # Calculate days since signup
    reference_date = pd.Timestamp('2024-12-31')
    customers_df['days_since_signup'] = (reference_date - customers_df['SignupDate']).dt.days
    
    # One-hot encode region
    region_dummies = pd.get_dummies(customers_df['Region'], prefix='region')
    
    # Combine features
    features_df = customers_df[['CustomerID', 'days_since_signup']].merge(
        customer_metrics, on='CustomerID', how='left'
    )
    features_df = features_df.merge(region_dummies, left_index=True, right_index=True)
    
    # Fill NaN values (customers with no transactions)
    features_df = features_df.fillna(0)
    
    return features_df

# Prepare features
features_df = prepare_features(customers_df, transactions_df)

# Scale the features
feature_columns = [col for col in features_df.columns if col != 'CustomerID']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df[feature_columns])

# Find optimal number of clusters using elbow method
inertias = []
db_scores = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)
    db_scores.append(davies_bouldin_score(scaled_features, kmeans.labels_))
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

# Plot evaluation metrics
plt.figure(figsize=(15, 5))

# Elbow curve
plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Davies-Bouldin scores
plt.subplot(1, 3, 2)
plt.plot(k_range, db_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Score')

# Silhouette scores
plt.subplot(1, 3, 3)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

# Select optimal number of clusters (you can adjust based on the plots)
optimal_k = 4  # This can be adjusted based on the evaluation metrics

# Perform final clustering
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = final_kmeans.fit_predict(scaled_features)

# Add cluster labels to the original data
features_df['Cluster'] = cluster_labels

# Analyze clusters
cluster_analysis = features_df.groupby('Cluster').agg({
    'transaction_count': 'mean',
    'total_spend': 'mean',
    'avg_transaction_value': 'mean',
    'days_since_signup': 'mean'
}).round(2)

print("\nCluster Analysis:")
print(cluster_analysis)

print("\nDavies-Bouldin Score:", davies_bouldin_score(scaled_features, cluster_labels))
print("Silhouette Score:", silhouette_score(scaled_features, cluster_labels))

# Visualize clusters
plt.figure(figsize=(12, 6))

# Scatter plot: Total Spend vs Transaction Count
plt.subplot(1, 2, 1)
scatter = plt.scatter(features_df['total_spend'], 
                     features_df['transaction_count'],
                     c=features_df['Cluster'],
                     cmap='viridis')
plt.xlabel('Total Spend')
plt.ylabel('Transaction Count')
plt.title('Clusters: Total Spend vs Transaction Count')
plt.colorbar(scatter)

# Scatter plot: Average Transaction Value vs Days Since Signup
plt.subplot(1, 2, 2)
scatter = plt.scatter(features_df['avg_transaction_value'], 
                     features_df['days_since_signup'],
                     c=features_df['Cluster'],
                     cmap='viridis')
plt.xlabel('Average Transaction Value')
plt.ylabel('Days Since Signup')
plt.title('Clusters: Avg Transaction Value vs Days Since Signup')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# Save results
features_df.to_csv('customer_segments.csv', index=False)