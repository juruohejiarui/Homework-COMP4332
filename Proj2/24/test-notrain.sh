#!/bin/bash

for ctx_size in 5000 7500 10000 20000 30000 40000
do
    echo "Context size = $ctx_size"
    python ./predict.py --data_root ./data --device cuda --out_root ./data/ --max_context $ctx_size --scale
    python ./evaluate.py --pred_name predict-ctx.csv  > result-hjr-$ctx_size-scaled.txt

    python ./predict.py --data_root ./data --device cuda --out_root ./data/ --max_context $ctx_size
    python ./evaluate.py --pred_name predict-ctx.csv > result-hjr-$ctx_size.txt
done