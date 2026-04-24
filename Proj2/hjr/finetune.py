import os
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

from tabpfn.finetuning import FinetunedTabPFNRegressor, FinetunedTabPFNClassifier
from tabpfn.finetuning.train_util import get_checkpoint_path_and_epoch_from_output_dir
from sklearn.metrics import root_mean_squared_error, accuracy_score
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
    
    pred_path = os.path.join(model_path, "finetuned-pred.csv")

    columns = train_df.columns[-1]

    X_train = train_df.drop(columns=[columns])
    y_train = train_df[columns]

    X_test = test_df
    y_test = result_df[columns]

    print(len(X_train), len(y_train), len(X_test), len(y_test))

    metric = None
    opt_ctx_samples = [10000, 5000, 4000, 3000, 2000, 1000]
    
    for ctx_samples in opt_ctx_samples :
        try :
            print(f"Trying with ctx_samples={ctx_samples}...")
            if os.path.basename(model_path).startswith("reg") :
                model = FinetunedTabPFNRegressor(
                    device='cuda',
                    save_checkpoint_interval=None,
                    epochs=max_epoch,
                    early_stopping_patience=15,
                    validation_split_ratio=0.1,
                    random_state=42,
                    n_finetune_ctx_plus_query_samples=ctx_samples
                )
                metric = root_mean_squared_error
            else :
                model = FinetunedTabPFNClassifier(
                    device='cuda',
                    save_checkpoint_interval=None,
                    epochs=max_epoch,
                    early_stopping_patience=15,
                    validation_split_ratio=0.1,
                    random_state=42,
                    n_finetune_ctx_plus_query_samples=ctx_samples
                )
                metric = accuracy_score

            model = model.fit(X_train, y_train, output_dir=Path(model_path))
            break
        except Exception as e :
            print(f"Error with ctx_samples={ctx_samples}: {e}")
            continue

    y_pred = model.predict(X_test)
    rmse = metric(y_test, y_pred)

    chk_path, bst_epoch = get_checkpoint_path_and_epoch_from_output_dir(Path(model_path), len(X_train))

    print(f"Best checkpoint path: {chk_path}, Best epoch: {bst_epoch}")

    print(f"Metric of {data_path}: {rmse}")

    # save predictions
    pd.DataFrame(y_pred, columns=[columns]).to_csv(pred_path, index=False)

    return rmse

if __name__ == "__main__" :
    root_path = "../data/reg"
    root_model_path = "models"

    res = {}
    for item in os.listdir(root_path) :
        data_path = os.path.join(root_path, item)
        model_path = os.path.join(root_model_path, f"reg-{item}")
        rmse = train_and_test(data_path, model_path)

        res[item] = rmse

    datas = pd.DataFrame(list(res.items()), columns=['name', 'rmse'])
    datas.to_csv('result-reg.txt')

    root_path = "../data/cls"
    root_model_path = "models"

    res = {}
    for item in os.listdir(root_path) :
        data_path = os.path.join(root_path, item)
        model_path = os.path.join(root_model_path, f"cls-{item}")
        rmse = train_and_test(data_path, model_path)

        res[item] = rmse

    datas = pd.DataFrame(list(res.items()), columns=['name', 'acc'])
    datas.to_csv('result-cls.txt')

    # move prediction files to ../data/cls and ../data/reg
    for item in os.listdir("models") :
        if item.startswith("reg-") :
            pred_path = os.path.join("models", item, "finetuned-pred.csv")
            if os.path.exists(pred_path) :
                os.rename(pred_path, os.path.join("../data/reg", item[4:], f"predict-f.csv"))
        elif item.startswith("cls-") :
            pred_path = os.path.join("models", item, "finetuned-pred.csv")
            if os.path.exists(pred_path) :
                os.rename(pred_path, os.path.join("../data/cls", item[4:], f"predict-f.csv"))