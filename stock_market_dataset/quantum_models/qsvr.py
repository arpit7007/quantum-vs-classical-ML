import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def run_qsvr(num_qubits=5):
    # ===============================
    # 1. Load processed data
    # ===============================
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")

    # ===============================
    # 2. SUBSAMPLE (CRITICAL FIX)
    # ===============================
    MAX_SAMPLES = 150   # keeps runtime reasonable

    X_train = X_train[:MAX_SAMPLES]
    y_train = y_train[:MAX_SAMPLES]

    X_test  = X_test[:MAX_SAMPLES]
    y_test  = y_test[:MAX_SAMPLES]

    # ===============================
    # 3. Feature → qubit pooling
    # ===============================
    # shape: (samples, num_qubits, features_per_qubit)
    X_train = X_train.reshape(len(X_train), num_qubits, -1)
    X_test  = X_test.reshape(len(X_test), num_qubits, -1)

    # mean + std pooling (richer than mean alone)
    X_train = np.concatenate(
        [X_train.mean(axis=2), X_train.std(axis=2)],
        axis=1
    )
    X_test = np.concatenate(
        [X_test.mean(axis=2), X_test.std(axis=2)],
        axis=1
    )

    # effective qubits doubled
    num_qubits = num_qubits * 2

    # ===============================
    # 4. Quantum kernel (REDUCED DEPTH)
    # ===============================
    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=2        # reps=3 was too slow for iteration
    )

    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    print("Computing quantum kernel (train)...")
    K_train = quantum_kernel.evaluate(X_train)

    print("Computing quantum kernel (test)...")
    K_test = quantum_kernel.evaluate(X_test, X_train)

    # ===============================
    # 5. SVR on quantum kernel
    # ===============================
    model = SVR(
        kernel="precomputed",
        C=10.0,
        epsilon=0.01
    )
    model.fit(K_train, y_train)

    preds = model.predict(K_test)

    # ===============================
    # 6. Metrics
    # ===============================
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    return rmse, r2
