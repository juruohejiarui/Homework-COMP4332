#!/bin/bash

cd ./hjr

python ./svdpp_meta.py --st_norm --valid_out val_pred.csv --test_out prediction.csv

cd ../sirui

python ./generate_test_predictions.py --retrain-with-val

cd ../

python ./average.py ./hjr/prediction.csv ./sirui/prediction.csv --output ./test.csv
python ./average.py ./hjr/val_pred.csv ./sirui/val_pred.csv --output ./val_pred.csv 

python ./evaluate.py --pred ./val_pred.csv
