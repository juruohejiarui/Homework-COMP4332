#!/bin/bash

for ctx_size in 5000 7500 10000 20000 30000
do
    echo "Context size = $ctx_size"
    python ./predict.py --data_root ../data --device cuda --out_root ../data/ --max_context $ctx_size --scale
    cd ../ && python ./evaluate.py > result-hjr-$ctx_size-scaled.txt && cd ./hjr

    python ./predict.py --data_root ../data --device cuda --out_root ../data/ --max_context $ctx_size
    cd ../ && python ./evaluate.py > result-hjr-$ctx_size.txt && cd ./hjr
done