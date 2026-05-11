# Project 3 — Rating Prediction

## Task

Given a (user, item) pair, predict the user's **star rating** (a real number on the
1.0 – 5.0 scale). The primary evaluation metric is **Root Mean Squared Error
(RMSE)**.

## Dataset


All files live in `data/`:

| File              | Rows     | Columns                                       | Notes                                        |
|-------------------|----------|-----------------------------------------------|----------------------------------------------|
| `train.csv`      | ~52 K    | `ReviewerID, ProductID, Text, Summary, Star`  | training set with full review text           |
| `validation.csv`  | ~6.6 K   | `ReviewerID, ProductID, Star`                 | held-out for local evaluation                |
| `test.csv`  | ~6.6 K   | `ReviewerID, ProductID, Star`                 | `Star` is **all 0.0** — fill it in           |
| `product.json`    | ~6.7 K   | list of dicts (`ProductID, title, categories, features, description, store, price, ...`) | item metadata |

All ratings are floats in `1.0-5.0`. 


## Dependencies

```bash
pip install torch pandas numpy
```

## Baselines

We provide two reference implementations under `baselines/`. Both are
trained on `data/train.csv`, monitored on `data/validation.csv` for
early stopping, and produce predictions on `data/test.csv`. Each
script writes two CSVs in the project root:

### `baselines/ncf.py` — Neural Collaborative Filtering

```bash
python baselines/ncf.py
```

- **Architecture**: each user and each item gets a 16-d learnable
  embedding. The two embeddings are concatenated and passed through a
  3-layer MLP `[32, 16, 1]` with ReLU + dropout. The MLP output is
  added to a single learnable global-mean parameter (initialised to
  the training mean ~4.46) so the network only has to model the
  small residual around the mean. The final score is clamped to
  `[1.0, 5.0]`.
- **Loss**: MSE between predicted and actual `Star`.
- **Best validation RMSE**: **0.9999**

### `baselines/wide_deep.py` — Wide & Deep

```bash
python baselines/wide_deep.py
```

- **Wide branch (memorisation)**: a global bias plus a per-user and
  per-item learned scalar bias. This explicitly captures "how high a
  user tends to rate" and "how good an item tends to be". Heavy weight
  decay (`1e-2`) on the per-ID biases keeps them shrunk toward zero
  for users / items with few interactions.
- **Deep branch (generalisation)**: 16-d user / item embeddings,
  concatenated and passed through an MLP `[32, 16, 1]` with ReLU +
  dropout. The deep branch predicts the **residual** that the wide
  branch can't memorise.
- **Output**: `wide + deep`, clamped to `[1.0, 5.0]`. 
- **Best validation RMSE**: **0.9960**


## Evaluation

Use `evaluate.py` to compute RMSE on the validation set:

```bash
python evaluate.py --pred val_pred_ncf.csv
```

The script merges your predictions onto `data/validation.csv` by
`(ReviewerID, ProductID)`, clips predictions to `[1.0, 5.0]`, and
warns if any pair is missing. The same script is used by the TA on
the hidden test labels via `--truth`.


## Submission Format

Submit a single CSV named `prediction.csv` with exactly three columns:

```
ReviewerID,ProductID,Star
AGKAS...,B003LPTAYI,4.32
AGCI7...,B0040FJ27S,3.91
...
```

## Grading Rule (RMSE on hidden test set)

| Grade | What it means                                                    | Report                              | Test RMSE |
|-------|------------------------------------------------------------------|-------------------------------------|-----------|
| 60%   | submission only                                                  | submission                          | ≤ 1.0   |
| 80%   | an easy baseline that most students can outperform                           |detailed explanation           | ≤ 0.99    |
| 90%   | a competitive baseline that about half students can surpass                | detailed explanation and analysis    | ≤ 0.98    |
| 100%  | a very competitive baseline   | excellent visualization and analysis  | ≤ 0.97    |


## Submission Checklist

- [ ] `prediction.csv` with the exact format above.
- [ ] Code that reproduces your model (any framework / language).
- [ ] A 1–2 page report describing your approach, ablations, and final train and validation RMSE.
- [ ] All packaged as `groupNo.zip` and uploaded to Canvas.
