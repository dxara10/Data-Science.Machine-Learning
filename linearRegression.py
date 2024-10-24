from sklearn.linear_model import LinearRegression
import numpy as np

# Dados
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 4, 5, 6])

# Criar e treinar modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, y)

# Prever valores
X_novo = np.array([[6], [7]])
y_pred = modelo.predict(X_novo)

print("Valores previstos:", y_pred)
