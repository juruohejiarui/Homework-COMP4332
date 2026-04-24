import os
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

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

def train_and_test(data_path : str, model_path : str, max_epoch : int = 100) -> tuple[float, float] :
    bst_model_path = f"{model_path}_bst"
    train_df = pd.read_csv(os.path.join(data_path, train_file_name))
    test_df = pd.read_csv(os.path.join(data_path, test_file_name))
    result_df = pd.read_csv(os.path.join(data_path, result_file_name))
    
    pred1_path = os.path.join(model_path, "pred1.csv")
    pred2_path = os.path.join(model_path, "pred2.csv")

    columns = train_df.columns[-1]

    X_train = train_df.drop(columns=[columns])
    y_train = train_df[columns]

    X_test = test_df
    y_test = result_df[columns]

    print(len(X_train), len(y_train), len(X_test), len(y_test))

    model = FinetunedTabPFNClassifier(
        device='cuda',
        save_checkpoint_interval=None,
        epochs=max_epoch,
        early_stopping_patience=8,
        validation_split_ratio=0.1,
        n_finetune_ctx_plus_query_samples=len(X_train)
    )
    model = model.fit(X_train, y_train, output_dir=Path(model_path))

    y_pred1 = model.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred1)

    chk_path, bst_epoch = get_checkpoint_path_and_epoch_from_output_dir(Path(model_path), len(X_train))

    print(f"Best checkpoint path: {chk_path}, Best epoch: {bst_epoch}")

    model = FinetunedTabPFNClassifier(
        device='cuda',
        save_checkpoint_interval=None,
        epochs=bst_epoch,
        early_stopping=False,
        n_finetune_ctx_plus_query_samples=len(X_train),
    )

    model = model.fit(X_train, y_train, X_train[0 : 4].copy(), y_train[0 : 4].copy(), output_dir=Path(bst_model_path))

    y_pred2 = model.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred2)

    print(f"ACC of {data_path}: {acc1} {acc2}")

    # save predictions
    pd.DataFrame(y_pred1, columns=[columns]).to_csv(pred1_path, index=False)
    pd.DataFrame(y_pred2, columns=[columns]).to_csv(pred2_path, index=False)

    return acc1, acc2

if __name__ == "__main__" :
    root_path = "../data/cls"
    root_model_path = "models"

    res = {}
    for item in os.listdir(root_path) :
        data_path = os.path.join(root_path, item)
        model_path = os.path.join(root_model_path, f"cls-{item}")
        acc1, acc2 = train_and_test(data_path, model_path)

        res[item] = (acc1, acc2)

    datas = pd.DataFrame(list(res.items()), columns=['name', 'rmse'])
    datas.to_csv('result-cls.csv')