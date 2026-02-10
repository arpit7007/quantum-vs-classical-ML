import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from stock_market_dataset.preprocessing.load_data import load_ohlcv_data
from stock_market_dataset.preprocessing.feature_engineering import create_lagged_features


def prepare_dataset(
    csv_path = "finace_dataset/data.csv",
    window=3,
    test_size=0.2
):
    df = load_ohlcv_data(csv_path)
    X, y = create_lagged_features(df, window=window,H=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    return X_train, X_test, y_train, y_test
