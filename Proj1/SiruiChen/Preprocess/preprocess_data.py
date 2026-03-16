import argparse
import os
from pathlib import Path

import pandas as pd

from .Data_augmentation import DataAugmentation


def load_raw_splits(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "valid.csv")
    test = pd.read_csv(data_dir / "test_no_label.csv")
    return train, valid, test


def apply_augmentation(train_df: pd.DataFrame, num_aug: int = 1) -> pd.DataFrame:
    """
    使用 EDA 对训练集做简单的数据增强。
    默认对每条样本做 num_aug 轮增强，每轮产生 2 条样本（同义词替换 + 随机插入）。
    """
    augmenter = DataAugmentation()
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(train_df["text"], train_df["label"]):
        augmented_pairs = augmenter.augment_text(str(text), int(label), num_aug=num_aug)
        for t, y in augmented_pairs:
            augmented_texts.append(t)
            augmented_labels.append(y)

    aug_df = pd.DataFrame({"text": augmented_texts, "label": augmented_labels})
    # 保留原始数据 + 增强数据
    combined = pd.concat([train_df, aug_df], ignore_index=True)
    return combined


def save_processed_splits(output_dir: Path, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)
    test_df.to_csv(output_dir / "test_no_label.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("..", "..", "data"),
        help="原始数据所在目录（包含 train.csv, valid.csv, test_no_label.csv）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("..", "Processed_data"),
        help="预处理后数据输出目录",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="是否对训练集做 EDA 数据增强",
    )
    parser.add_argument(
        "--num_aug",
        type=int,
        default=1,
        help="对每条训练样本做几轮增强（每轮 2 条增强样本）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"读取原始数据：{data_dir}")
    train_df, valid_df, test_df = load_raw_splits(data_dir)

    if args.use_augmentation:
        print("对训练集进行数据增强（EDA）...")
        train_df = apply_augmentation(train_df, num_aug=args.num_aug)
        print(f"增强后训练集大小：{len(train_df)} 条样本")

    print(f"保存预处理后的数据到：{output_dir}")
    save_processed_splits(output_dir, train_df, valid_df, test_df)
    print("预处理完成。")


if __name__ == "__main__":
    main()

