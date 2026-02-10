import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def run_svr():
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    model = SVR(kernel="rbf", C=100, gamma="scale")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    return rmse, r2
