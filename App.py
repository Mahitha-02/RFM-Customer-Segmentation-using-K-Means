import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("sales.csv")


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


df['Date'] = pd.to_datetime(df['Date'])


df['Total'] = df['Quantity'] * df['Price']


reference_date = df['Date'].max() + pd.Timedelta(days=1)


rfm = df.groupby('CustomerID').agg({
    'Date': lambda x: (reference_date - x.max()).days,
    'CustomerID': 'count',
    'Total': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']


scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)


kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)


plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Segmentation")
plt.show()


rfm.to_csv("rfm_output.csv", index=False)
