from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

NDArray = np.ndarray[Any, Any]

np.set_printoptions(precision=4, suppress=True)
np.random.seed(357)

def load(path: str) -> tuple[NDArray, NDArray]:
    data = pd.read_csv(path)
    y = data["cena"].to_numpy()
    x = data.loc[:, data.columns != "cena"].to_numpy()
    return x, y

x_train, y_train = load("mieszkania.csv")
x_test, y_test = load("mieszkania_test.csv")

label_encoder = LabelEncoder()
label_encoder.fit(x_train[:, 1])
x_train[:, 1] = label_encoder.transform(x_train[:, 1])
x_test[:, 1] = label_encoder.transform(x_test[:, 1])

x_train = x_train.astype(np.float64)
x_test = x_test.astype(np.float64)

def mse(ys: NDArray, ps: NDArray) -> np.float64:
    assert ys.shape == ps.shape
    return np.mean((ys - ps) * (ys - ps))

def train(
    X: NDArray, y: NDArray, alpha: float = 1e-7, n_iterations: int = 100000
) -> tuple[NDArray, np.float64]:
    
    B, F = X.shape
    w = np.random.uniform(size=(F,), low=-1/np.sqrt(F), high=1/np.sqrt(F))
    b = np.mean(y)
    
    for it in range(n_iterations):
        preds = X @ w + b
        
        gradients = np.zeros(F)
        for i in range(F):
            gradients[i] = -2.0 * np.mean((y - preds) * X[:, i])
        grad_b = -2.0 * np.mean(y-preds)

        w = w - alpha * gradients
        b = b - alpha * grad_b
    return w, b

w, b = train(x_train, y_train, 0.001, 5000)
preds_test = x_test @ w + b
print("test MSE:", mse(y_test, preds_test))

# def msle(ys: NDArray, ps: NDArray) -> np.float64:
#     assert ys.shape == ps.shape
#     #################################
#     # TODO: Implement this function #
#     #################################

# """## Linear regression (standard)

# Now, let's implement training of a standard linear regression model via gradient descent.
# """

# def train(
#     x: NDArray, y: NDArray, alpha: float = 1e-7, n_iterations: int = 100000
# ) -> tuple[NDArray, np.float64]:
#     """Linear regression (which optimizes MSE). Returns (weights, bias)."""

#     # B is batch size (number of observations).
#     # F is number of (input) features.
#     B, F = x.shape
#     assert y.shape == (B,)

#     # TODO #

# weights, bias = train(x_train, y_train)
# preds_test = ... # TODO #
# print("test MSLE:", msle(y_test, preds_test))

# """## Linear regression (MSLE)

# Note that the loss function that the algorithms optimizes (i.e $MSE$) differs from $MSLE$. We've already seen that this may result in a suboptimal solution.

# How can you change the setting so that we optimze $MSLE$ instead?

# Hint:
# <sub><sup><sub><sup><sub><sup>
# Be lazy. We don't want to change the algorithm.
# Use the chain rule and previous computations to get formulas for the gradient.
# </sup></sub></sup></sub></sup></sub>
# """

# def train_msle(
#     x: NDArray, y: NDArray, alpha: float = 1e+4, n_iterations: int = 50000
# ) -> tuple[NDArray, NDArray]:
#     """Gradient descent for MSLE."""

#     #############################################
#     # TODO: Optimize msle and compare the error #
#     #############################################


# weights, bias = train_msle(x_train, y_train)
# preds_test = ... # TODO #
# print("test MSLE: ", msle(y_test, preds_test))
