import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')

x_train = data['studytime'].to_numpy(dtype=float)
y_train = data['score'].to_numpy(dtype=float)

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -2/n * x * (y - (m_now*x + b_now))
        b_gradient += -2/n * (y - (m_now*x + b_now))
    
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return m, b

m = 0
b = 0
L = 0.001
epochs = 300

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    if i % 50 == 0: print(f"LosN: {mse(y_train, m * x_train + b)}")

print(m,b)

plt.scatter(data.studytime, data.score, color="black")
plt.plot(list(range(0, 12)), [m*x+b for x in range(0,12)], color="red")
plt.show()