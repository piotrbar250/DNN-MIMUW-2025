#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Szablon: regresja liniowa (własny trening) + MSE/RMSE.
Uzupełnij wyłącznie funkcję `train(X, y, alpha, n_iterations)`,
która ma zwrócić: (weights: np.ndarray[k], bias: float).
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------
# 1) TU UZUPEŁNIASZ: Twój trening. Ma zwrócić (weights, bias)
#    - X: ndarray shape (n_samples, n_features)
#    - y: ndarray shape (n_samples,)
#    - alpha: float (np. learning rate)
#    - n_iterations: int (np. liczba iteracji GD)
# ------------------------------------------------------------
def train(X, y, alpha=1e-2, n_iterations=2000):
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

# ------------------------------------------------------------
# 2) Preprocessing danych
# ------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """
    - One-hot dla 'dzielnica' (drop_first=True).
    - Bez skalowania.
    Zwraca: X (ndarray), y (ndarray), feature_names (lista)
    """
    if "cena" not in df.columns:
        raise ValueError("Brakuje kolumny docelowej 'cena'.")

    df_enc = pd.get_dummies(df, columns=["dzielnica"], drop_first=True)
    feature_names = [c for c in df_enc.columns if c != "cena"]
    X = df_enc[feature_names].to_numpy(dtype=float)
    y = df_enc["cena"].to_numpy(dtype=float)
    return X, y, feature_names


# ------------------------------------------------------------
# 3) Predykcja na bazie (weights, bias)
# ------------------------------------------------------------
def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return X @ weights + bias


# ------------------------------------------------------------
# 4) Ewaluacja
# ------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "eval"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"[{label}] MSE: {mse:,.2f}")
    print(f"[{label}] RMSE: {rmse:,.2f}")
    return mse, rmse


# ------------------------------------------------------------
# 5) Główna ścieżka uruchomienia
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Regresja liniowa z własnym treningiem i MSE/RMSE")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Ścieżka do CSV z kolumnami: m2,dzielnica,ilość_sypialni,ilość_łazienek,rok_budowy,parking_podziemny,cena",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.0,
        help="Udział zbioru testowego (0.0 = bez podziału).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Ziarno losowe do podziału train/test.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-6,
        help="Współczynnik uczenia (alpha) przekazywany do train().",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10_000,
        help="Liczba iteracji przekazywana do train().",
    )
    args = parser.parse_args()

    # Wczytanie i preprocessing
    df = pd.read_csv(args.csv)
    X, y, feat_names = preprocess(df)

    # Podział (opcjonalnie)
    if args.test_size and args.test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        # Trening
        weights, bias = train(X_train, y_train, args.alpha, args.n_iter)

        # Ewaluacja
        print("== Wyniki ==")
        evaluate(y_train, predict(X_train, weights, bias), label="train")
        evaluate(y_test, predict(X_test, weights, bias), label="test")
    else:
        # Trening i ewaluacja in-sample
        weights, bias = train(X, y, args.alpha, args.n_iter)
        print("== Wyniki (in-sample) ==")
        evaluate(y, predict(X, weights, bias), label="full")

    # Informacyjnie: nazwy cech i wagi (jeśli chcesz je obejrzeć)
    print("\nCechy (kolejność w X):")
    for i, n in enumerate(feat_names):
        print(f"{i:2d}. {n}")
    print("\n(Uwaga: wartości wag zależą od Twojej implementacji train().)")

if __name__ == "__main__":
    main()
