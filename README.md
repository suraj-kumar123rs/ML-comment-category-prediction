# Comment Category Prediction

Multi-class classification of user comments into 4 categories, built for the **IIT Madras ML Kaggle competition** (Comment Category Prediction Challenge).

**Result:** a soft-voting ensemble of Logistic Regression, XGBoost and Linear SVM reached **0.819 macro F1** on validation and **0.816 macro F1** on the Kaggle leaderboard.

📓 **Notebook:** [`comment_category_prediction.ipynb`](comment_category_prediction.ipynb)
---

## Problem

Each row is a user comment on a post, along with metadata about it. The task is to predict its category (`label` 0–3). The competition metric is **macro F1**, so every class counts equally, including the rare ones.

| | Rows | Columns |
|---|---|---|
| Train | 198,000 | 15 |
| Test | 102,000 | 14 |

**Features available**
- `comment` — the comment text
- Engagement: `upvote`, `downvote`, `emoticon_1/2/3`
- Other numeric: `post_id`, `if_1`, `if_2`
- Categorical: `race`, `religion`, `gender` (~73% missing), `disability` (boolean)
- `created_date`

## Key finding from EDA: heavy class imbalance

| Label | Count | Share |
|---|---|---|
| 0 | 114,173 | 57.7% |
| 2 | 62,440 | 31.5% |
| 1 | 15,918 | 8.0% |
| 3 | 5,469 | 2.8% |

Label 3 is under 3% of the data. Since the metric is macro F1, a model that ignores it gets heavily penalised, so every model below uses `class_weight="balanced"` where available.

Other EDA in the notebook: comment length distribution (mostly short comments, long tail), comment length vs label, emoticon usage, and a correlation heatmap of the numeric features (low multicollinearity; upvotes and downvotes moderately correlated).

## Approach

**1. Preprocessing** — one scikit-learn `ColumnTransformer`, fit on train and applied to test:

| Input | Treatment |
|---|---|
| `comment` | TF-IDF: unigrams + bigrams, 30k features, English stop words removed, `min_df=5`, sublinear TF |
| `race`, `religion`, `gender` | Missing → `"Unknown"`, then one-hot encoded |
| Numeric + `disability` | Standard scaled |
| Engineered | `comment_length` (character count) |

Final feature matrix: **30,032 sparse features**. Validation: stratified 80/20 split.

**2. Models**

| Model | Validation macro F1 |
|---|---|
| Logistic Regression (`class_weight="balanced"`) | 0.793 |
| Logistic Regression, grid-searched `C` ∈ {0.1, 1, 10} | 0.792 (best `C=1`, same as default) |
| XGBoost (400 trees, depth 6, lr 0.1) | 0.785 |
| Linear SVM (`class_weight="balanced"`) | 0.687 |
| **Ensemble: LR + XGBoost + SVM (averaged probabilities)** | **0.819** |

**3. Ensemble** — Linear SVM has no `predict_proba`, so its decision scores were converted to probabilities with softmax. The three probability vectors were averaged and the argmax taken.

The models make different mistakes, which is why averaging them helps. On the rarest class (label 3), Logistic Regression had high recall but low precision (0.76 / 0.48), while XGBoost was the opposite (0.42 / 0.75). The ensemble balanced both at 0.65 / 0.65, lifting label-3 F1 from 0.59 (LR) to 0.65.

## Results

| | Macro F1 |
|---|---|
| Best single model (Logistic Regression) | 0.793 |
| Ensemble — validation | **0.819** |
| Ensemble — Kaggle leaderboard | **0.816** |

The small gap between validation and leaderboard suggests the model generalises well to unseen data.

## What I'd try next

- Tune the ensemble weights instead of a plain average
- Per-class threshold tuning for label 3
- Fine-tune a small transformer (e.g. DistilBERT) on the comment text and add it to the ensemble

## Run it

The notebook was built on Kaggle. To run it locally:

```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn
```

Download `train.csv` and `test.csv` from the competition, then update the two `pd.read_csv(...)` paths in the notebook.

**Tech:** Python · pandas · scikit-learn · XGBoost · SciPy · Matplotlib · Seaborn

---

**Suraj Kumar** · BS Data Science, IIT Madras
[GitHub](https://github.com/suraj-kumar123rs) · [LinkedIn](https://www.linkedin.com/in/suraj-kumar-rs/) · surajkumar805122@gmail.com
