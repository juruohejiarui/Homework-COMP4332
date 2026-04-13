from tabpfn.finetuning import FinetunedTabPFNClassifier
from sklearn.metrics import accuracy_score
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

    model = FinetunedTabPFNClassifier(
        device='cuda',
        save_checkpoint_interval=None,
        epochs=40,
        early_stopping_patience=15,
    )
    model.fit(X_train, y_train, X_test, y_test, output_dir=Path(model_path))


    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy of {data_path}: {acc}")

    return acc


if __name__ == "__main__" :
    root_path = "../data/cls"
    root_model_path = "models"

    res = {}
    for item in os.listdir(root_path) :
        data_path = os.path.join(root_path, item)
        model_path = os.path.join(root_model_path, f"cls-{item}")
        acc = train_and_test(data_path, model_path)

        res[item] = acc

    datas = pd.DataFrame(list(res.items()), columns=['name', 'acc'])
    datas.to_csv('result-cls.txt')