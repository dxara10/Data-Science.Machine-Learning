from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carregar conjunto de dados Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Criar e treinar modelo SVM
modelo = SVC(kernel='linear', C=1.0)
modelo.fit(X_train, y_train)

# Avaliar modelo
score = modelo.score(X_test, y_test)
print("Acurácia do modelo SVM:", score)
