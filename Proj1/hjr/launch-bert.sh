#!/bin/bash
# python bert.py --train_csv ../data/train.csv --valid_csv ../data/valid.csv --epochs 3 --batch_size 64 --save_path my_best_model.pt

num=0
while (( $num < 20 ))
do
    echo "robertq-large $num"
    python bert.py --train_csv ../data/train.csv --valid_csv ../data/valid.csv --epochs 3 --lr 1e-5 --batch_size 64 --save_path best_robert.pt --model_name ~/Documents/Resources/Models/Chat/roberta-large
    sleep 2m
    (( num = $num + 1 ))
done
