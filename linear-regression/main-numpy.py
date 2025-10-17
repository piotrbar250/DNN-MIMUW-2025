import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')

x_train = data['studytime'].to_numpy(dtype=float)
y_train = data['score'].to_numpy(dtype=float)

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def train_mse(x, y, alpha, n_iterations):
    m = 0
    b = 0
    for i in range(n_iterations):
        print(f"Loss: {mse(y, m * x + b)}")
        m_gradient = np.mean(-2 * x * (y - (m * x + b)))
        b_gradient = np.mean(-2 * (y - (m * x + b)))
        m = m - alpha * m_gradient
        b = b - alpha * b_gradient
    return m, b

m, b = train_mse(x_train, y_train, 0.001, 300)
print(m, b)

plt.scatter(data.studytime, data.score, color="black")
plt.plot(list(range(0, 12)), [m*x+b for x in range(0,12)], color="red")
plt.show()
