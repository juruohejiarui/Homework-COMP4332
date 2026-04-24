import os
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

from tabpfn import TabPFNClassifier
from tabpfn.finetuning import FinetunedTabPFNClassifier
from tabpfn.finetuning.train_util import get_checkpoint_path_and_epoch_from_output_dir
from sklearn.metrics import accuracy_score
from pathlib import Path
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

train_file_name = "train.csv"
test_file_name = "test.csv"
result_file_name = "result.csv"

def train_and_test(data_path : str, model_path : str, max_epoch : int = 100) -> float :
    bst_model_path = f"{model_path}_bst"
    train_df = pd.read_csv(os.path.join(data_path, train_file_name))
    test_df = pd.read_csv(os.path.join(data_path, test_file_name))
    result_df = pd.read_csv(os.path.join(data_path, result_file_name))

    columns = result_df.columns[0]

    X_train = train_df.drop(columns=[columns])
    y_train = train_df[columns]

    X_test = test_df
    y_test = result_df[columns]

    print(len(X_train), len(y_train), len(X_test), len(y_test))

    model = FinetunedTabPFNClassifier(
        device='cuda',
        save_checkpoint_interval=None,
        epochs=max_epoch,
        early_stopping_patience=15,
        validation_split_ratio=0.1,
        n_finetune_ctx_plus_query_samples=len(X_train)
    )
    model = model.fit(X_train, y_train, output_dir=Path(model_path))

    chk_path, bst_epoch = get_checkpoint_path_and_epoch_from_output_dir(Path(model_path), len(X_train))

    print(f"Best checkpoint path: {chk_path}, Best epoch: {bst_epoch}")

    model = FinetunedTabPFNClassifier(
        device='cuda',
        save_checkpoint_interval=None,
        epochs=bst_epoch,
        early_stopping=False,
        n_finetune_ctx_plus_query_samples=len(X_train)
    )

    indices = X_train.groupby(y_train).sample(n=1, random_state=42).index
    X_sub = X_train.loc[indices]
    y_sub = y_train.loc[indices]

    model = model.fit(X_train, y_train, X_sub, y_sub, output_dir=Path(bst_model_path))

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"ACC of {data_path}: {acc}")

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

    datas = pd.DataFrame(list(res.items()), columns=['name', 'rmse'])
    datas.to_csv('result-cls.csv')