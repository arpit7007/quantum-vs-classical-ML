from stock_market_dataset.preprocessing.prepare_datasets import prepare_dataset
from stock_market_dataset.classical_models.svr_rbf import run_svr
from stock_market_dataset.classical_models.linear_regression import run_linear
from stock_market_dataset.quantum_models.qsvr import run_qsvr
from stock_market_dataset.evaluation.metrics import print_results


# Step 1: Prepare dataset
prepare_dataset(
    csv_path="finace_dataset/data.csv",
    window=7
)

# Step 2: Classical models
rmse_lr, r2_lr = run_linear()
print_results("Linear Regression", rmse_lr, r2_lr)

rmse_svr, r2_svr = run_svr()
print_results("SVR (RBF)", rmse_svr, r2_svr)

# Step 3: Quantum model
rmse_qsvr, r2_qsvr = run_qsvr(num_qubits=5)
print_results("QSVR (5-Qubit)", rmse_qsvr, r2_qsvr)
