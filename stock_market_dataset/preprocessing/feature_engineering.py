import numpy as np

def create_lagged_features(df, window=3, H=3):
    features = ["Open", "High", "Low", "Close", "Volume"]

    X, y = [], []

    # 🔥 stop early so i + H never goes out of bounds
    for i in range(window, len(df) - H):
        row = []

        for w in range(1, window + 1):
            for f in features:
                row.append(df.loc[i - w, f])

        X.append(row)

        # log-return over horizon H
        y.append(np.log(df.loc[i + H, "Close"] / df.loc[i, "Close"]))

    return np.array(X), np.array(y)
