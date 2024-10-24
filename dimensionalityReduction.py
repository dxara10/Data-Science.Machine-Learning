from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Carregar conjunto de dados Iris
iris = load_iris()
X = iris.data

# Aplicar PCA para redução de dimensionalidade
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Dimensões reduzidas com PCA:", X_pca.shape)
