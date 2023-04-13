from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

iris_df = datasets.load_iris()

pca = PCA(2)

X, y = iris_df.data, iris_df.target
X_proj = pca.fit_transform(X)

plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
plt.show()

print(pca.explained_variance_ratio_)

# Set the size of the plot
plt.figure(figsize=(10, 4))

# create color map
colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow', 'green', 'red'])

k = 3
# running kmeans clustering into three clusters
# random state set to 0 so we get the same result each time (deterministic)
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X_proj)

labels = kmeans.labels_
clusters = kmeans.cluster_centers_
print(clusters)

# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[y], s=40)
plt.title('Real Classification')

# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[labels], s=40)
plt.title('K Means Classification')

plt.show()
