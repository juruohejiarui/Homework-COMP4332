#!/bin/bash

cd ./svdpp_meta

python ./svdpp_meta.py --st_norm --valid_out val_pred.csv --test_out prediction.csv

cd ../lightgbm

python ./generate_test_predictions.py

cd ../

python ./average.py ./svdpp_meta/prediction.csv ./lightgbm/prediction.csv --output ./test.csv
python ./average.py ./svdpp_meta/val_pred.csv ./lightgbm/val_pred.csv --output ./val_pred.csv 

python ./evaluate.py --pred ./val_pred.csv
