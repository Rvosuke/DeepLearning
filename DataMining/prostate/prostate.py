import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    data = pd.read_csv(file_path, delimiter="\t")
    data = data.drop("Unnamed: 0", axis=1)
    return data


def preprocess_data(data):
    train_data = data[data["train"] == "T"].drop("train", axis=1)
    test_data = data[data["train"] == "F"].drop("train", axis=1)

    X_train = train_data.drop("lpsa", axis=1)
    y_train = train_data["lpsa"]
    X_test = test_data.drop("lpsa", axis=1)
    y_test = test_data["lpsa"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X, y, cv=10):
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    return -np.mean(scores)


def main():
    # 读取数据
    data = load_data("prostate.data")

    # 预处理数据
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(data)
    scoring = {'mse': 'neg_mean_squared_error', 'r2': 'r2'}

    # 模型训练与评估
    models_and_params = [
        ('LinearRegression', LinearRegression(), {}),
        ('SGDRegressor', SGDRegressor(max_iter=1000, tol=1e-3), {
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'eta0': [0.001, 0.01, 0.1],
            'max_iter': [500, 1000, 2000],
            'alpha': [0.0001, 0.001, 0.01]
        }),
        ('Ridge', Ridge(), {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        })
    ]

    # 超参数优化
    X_full = np.concatenate((X_train_scaled, X_test_scaled), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42
    )

    best_model = None
    best_params = None
    best_mse = float("inf")
    best_r2 = -float("inf")

    for name, model, params in models_and_params:
        print(f"Searching for best parameters of {name}...")
        grid_search = GridSearchCV(model, params, cv=10, scoring=scoring, refit='mse', return_train_score=True)
        grid_search.fit(X_train_full, y_train_full)

        mse = -grid_search.cv_results_['mean_test_mse'][grid_search.best_index_]
        r2 = grid_search.cv_results_['mean_test_r2'][grid_search.best_index_]
        print(f"{name} - Best MSE: {mse:.4f}, Best R^2: {r2:.4f}, Best parameters: {grid_search.best_params_}")

        if mse < best_mse:
            best_mse = mse
            best_r2 = r2
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

    print(
        f"Best model: {best_model}, Best parameters: {best_params}, Best MSE: {best_mse:.4f}, Best R^2: {best_r2:.4f}")


if __name__ == "__main__":
    main()
