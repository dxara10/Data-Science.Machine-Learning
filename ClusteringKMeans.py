from sklearn.cluster import KMeans
import numpy as np

# Dados
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Criar e treinar modelo K-Means
modelo = KMeans(n_clusters=2)
modelo.fit(X)

# Centróides
print("Centróides:", modelo.cluster_centers_)
