
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Input data
df = pd.read_csv("prediction_error_data_1094.csv")  # 确保文件在同目录下
data = df["value"].values.reshape(-1, 1)
# Operation
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(data)
centers = kmeans.cluster_centers_
# Save
df["cluster"] = labels
df.to_csv("clustered_data_10groups.csv", index=False)
print("Save as clustered_data.csv")
# Plot
plt.figure(figsize=(10, 4))
plt.scatter(range(len(data)), data.flatten(), c=labels, cmap='tab10', s=10, label='Data point')
plt.scatter(
    [np.where(labels == i)[0].mean() for i in range(10)],
    centers.flatten(),
    marker='x', color='black', s=100, label='center')
plt.title("K-Means results（K=10）")
plt.xlabel("K")
plt.ylabel("value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
