import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Configs
DATA_PATH = Path("datasets/Final_Resale_Data.csv")
MODEL_PATH = Path("models/test_lgbm_model")
CATEGORICAL_COLS = ["Flat_Type", "Within_1km_of_Pri", "Mature"]

# Main Function
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    
    # Load Dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    # Convert to Categorical Column
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    # Create Train and Test Set (with Log Transformation of Price Label)
    X = df.drop("Price", axis=1)
    y = np.log1p(df["Price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Dataset for Light GBM Model Fitting
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=CATEGORICAL_COLS)

    # Light GBM Hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "max_depth": 5,
        "learning_rate": 0.05,
        "lambda_l2": 2,
        "min_child_samples": 40,
        "bagging_fraction": 0.7,
        "verbose": -1,
        "random_state": 42,
        "monotone_constraints": [0, 0, 0, 1, 0, 0, 0, 0, 0]
    }

    # Train Model
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=30000,
        valid_sets=[train_data, test_data],
        valid_names=["train", "test"],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    # Save Model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH.with_suffix(".pkl"), "wb") as f:
        pickle.dump(bst, f)
    print(f"Model saved to {MODEL_PATH}")

    # Evaluation
    y_test_pred_log = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_train_pred_log = bst.predict(X_train, num_iteration=bst.best_iteration)
    test_pred = np.expm1(y_test_pred_log)
    train_pred = np.expm1(y_train_pred_log)
    y_test_exp = np.expm1(y_test)
    y_train_exp = np.expm1(y_train)
    print("Test RMSE:", root_mean_squared_error(y_test_exp, test_pred))
    print("Train RMSE:", root_mean_squared_error(y_train_exp, train_pred))
    test_mape  = np.mean(np.abs((y_test_exp  - test_pred ) / y_test_exp ))  * 100
    train_mape = np.mean(np.abs((y_train_exp - train_pred) / y_train_exp))  * 100

    print(f"Test  MAPE: {test_mape:.2f}%")
    print(f"Train MAPE: {train_mape:.2f}%")

    # Feature Importance
    importances = bst.feature_importance(importance_type='gain')
    features = bst.feature_name()
    for feature, score in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.2f}")

if __name__ == "__main__":
    main()