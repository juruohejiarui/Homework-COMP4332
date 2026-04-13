from tabpfn.finetuning import FinetunedTabPFNRegressor
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
import os
import pandas as pd

train_file_name = "train.csv"
test_file_name = "test.csv"
result_file_name = "result.csv"

def train_and_test(data_path : str, model_path : str) :
    train_df = pd.read_csv(os.path.join(data_path, train_file_name))
    test_df = pd.read_csv(os.path.join(data_path, test_file_name))
    result_df = pd.read_csv(os.path.join(data_path, result_file_name))

    columns = result_df.columns[0]

    X_train = train_df.drop(columns=[columns])
    y_train = train_df[columns]

    X_test = test_df
    y_test = result_df[columns]

    print(len(X_train), len(y_train), len(X_test), len(y_test))

    model = FinetunedTabPFNRegressor(
        device='cuda',
        save_checkpoint_interval=None,
        epochs=40,
        early_stopping_patience=15,
        validation_split_ratio=0.1
    )
    model = model.fit(X_train, y_train, output_dir=Path(model_path))

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"RMSE of {data_path}: {rmse}")

    return rmse

if __name__ == "__main__" :
    root_path = "../data/reg"
    root_model_path = "models"

    res = {}
    for item in os.listdir(root_path) :
        data_path = os.path.join(root_path, item)
        model_path = os.path.join(root_model_path, f"reg-{item}")
        acc = train_and_test(data_path, model_path)

        res[item] = acc

    datas = pd.DataFrame(list(res.items()), columns=['name', 'rmse'])
    datas.to_csv('result-reg.txt')