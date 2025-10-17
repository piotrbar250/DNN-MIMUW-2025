import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load(path: str):
    data = pd.read_csv(path)
    y = data["cena"].to_numpy(dtype=np.float64)
    X = data.loc[:, data.columns != "cena"].copy()
    return X, y

X_train_df, y_train = load("mieszkania.csv")
X_test_df,  y_test  = load("mieszkania_test.csv")

# 1) Kategoryczna dzielnica – NA RAZIE zostańmy przy LabelEncoder (minimalna zmiana)
le = LabelEncoder()
X_train_df["dzielnica"] = le.fit_transform(X_train_df["dzielnica"])
X_test_df["dzielnica"]  = le.transform(X_test_df["dzielnica"])  # zadziała tylko, jeśli w test nie ma nowych dzielnic

# 2) Standaryzacja cech numerycznych
num_cols = ["m2","ilość_sypialni","ilość_łazienek","rok_budowy","parking_podziemny"]
scaler = StandardScaler()
X_train_df[num_cols] = scaler.fit_transform(X_train_df[num_cols])
X_test_df[num_cols]  = scaler.transform(X_test_df[num_cols])

X_train = X_train_df.to_numpy(dtype=np.float64)
X_test  = X_test_df.to_numpy(dtype=np.float64)

def mse(y, p): return np.mean((y - p)**2)

# def train(X, y, alpha=1e-2, n_iterations=2000):
#     B, F = X.shape
#     rng = np.random.default_rng(357)
#     w = rng.uniform(low=-1/np.sqrt(F), high=1/np.sqrt(F), size=F)
#     b = np.mean(y)

#     for i in range(n_iterations):
#         preds = X @ w + b

#         if i % 50 == 0:
#             print(f"Loss: {mse(y, preds)}")

#         err = y - preds
#         # wektorowo:
#         grad_w = -2.0/B * (X.T @ err)
#         grad_b = -2.0/B * np.sum(err)
#         w -= alpha * grad_w
#         b -= alpha * grad_b
#     return w, b

def train(X, y, alpha=1e-2, n_iterations=2000):
    B, F = X.shape
    w = np.random.uniform(size=(F,), low=-1/np.sqrt(F), high=1/np.sqrt(F))
    b = np.mean(y)
    
    for it in range(n_iterations):
        preds = X @ w + b
        x
        gradients = np.zeros(F)
        for i in range(F):
            gradients[i] = -2.0 * np.mean((y - preds) * X[:, i])
        grad_b = -2.0 * np.mean(y-preds)

        w = w - alpha * gradients
        b = b - alpha * grad_b
    return w, b

w, b = train(X_train, y_train, alpha=1e-2, n_iterations=5000)
preds_train = X_train @ w + b
preds_test = X_test @ w + b

print("train MSE:", mse(y_train, preds_train))
print("train RMSE:", np.sqrt(mse(y_train, preds_train)))

print("test MSE:", mse(y_test, preds_test))
print("test RMSE:", np.sqrt(mse(y_test, preds_test)))