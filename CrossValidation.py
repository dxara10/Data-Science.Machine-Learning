from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Carregar conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Criar e avaliar modelo de regressão logística com cross-validation
modelo = LogisticRegression()
scores = cross_val_score(modelo, X, y, cv=5)

print("Acurácias do cross-validation:", scores)
