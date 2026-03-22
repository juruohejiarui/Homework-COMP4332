# 1. 预处理（可选择是否做 augment）
python -m Preprocess.preprocess_data `
  --data_dir ../data `
  --output_dir ./Processed_data `
  --use_augmentation `
  --num_aug 1

python -m Preprocess.preprocess_data `
  --data_dir ../data `
  --output_dir ./Processed_data `
  --num_aug 1

# 2. 训练三个模型 + ensemble（基于 Processed_data）
$env:HF_ENDPOINT = "https://hf-mirror.com" 
python -m training.run_hf_models_and_ensemble `
  --data_dir ./Processed_data `
  --output_root ./training_output `
  --epochs 3

python -m training.run_hf_models_and_ensemble `
  --data_dir ./Processed_data `
  --output_root ./training_output `
  --epochs 3 `
  --train_batch_size 64 `
  --eval_batch_size 128

# 3. 在 Proj1 根目录评估某个模型（或 ensemble 的 dev 预测，如果你之后也输出 dev 的 ensemble）
cd ..
python SiruiChen/evaluate_baseline_model.py `
  --pred SiruiChen/training_output/20260316_153045/bert_base_uncased/bert_base_uncased_valid_pred.csv