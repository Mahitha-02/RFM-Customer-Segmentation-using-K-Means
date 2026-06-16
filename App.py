import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("rfm_customer_data_enhanced.csv")


print("Dataset Preview:")
print(df.head())

print("\nColumns:")
print(df.columns)


df = df.dropna(subset=['CustomerID'])


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)


rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()


rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("\nRFM Table:")
print(rfm.head())


X = rfm[['Recency', 'Frequency', 'Monetary']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()


kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClustered Customers:")
print(rfm.head())


plt.figure(figsize=(8, 5))
plt.scatter(
    rfm['Frequency'],
    rfm['Monetary'],
    c=rfm['Cluster']
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.show()


rfm.to_csv("RFM_Segments.csv", index=False)

print("\nRFM Segmentation completed successfully!")
print("Output saved as: RFM_Segments.csv")
