# Online Shoppers Purchase Intention: model comparison and F1-oriented threshold tuning

This is a Binary classification problem, the purpose is to predict whether an e-commerce browsing session ends with a purchase (`Revenue = True`), comparing a majority-class baseline, Logistic Regression, Decision Tree, Random Forest and a Dense Neural Network. For every model the decision threshold is also tuned to **maximise the F1-score**.

Everything lives in a single notebook: `Definitivo.ipynb`.

---

## 1. Dataset

[Online Shoppers Purchasing Intention](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention/data) (Sakar et al., 2019, *Neural Computing and Applications*).

- 12,330 sessions, each from a different user over one year (avoids bias toward a specific campaign, special day, user profile or period).
- 17 features + 1 binary target (`Revenue`), no missing values.
- **Strong class imbalance**: 10,422 sessions without purchase vs 1,908 with purchase (about 15.5 %).

| Feature | Meaning |
|---|---|
| `Administrative`, `Informational`, `ProductRelated` | Number of pages of that type visited in the session |
| `*_Duration` | Seconds spent on pages of that type |
| `BounceRates` | Share of visitors who leave after a single interaction (Google Analytics) |
| `ExitRates` | Share of page views that were the last of the session (Google Analytics) |
| `PageValues` | Average value of the pages visited before a transaction (Google Analytics); 0 = none led to a transaction |
| `SpecialDay` | Closeness to a special day (e.g. Valentine's Day) |
| `Month`, `Weekend`, `VisitorType` | When the session happened / type of visitor |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Anonymous integer codes |
| **`Revenue`** | **Target**: the session ended with a purchase |

### Why accuracy is not enough
This dataset is highly unbalanced because about the 84.5% of the sessions ended without a purchase (this is visible from the cell 3.2 of the notebook), meaning that a model that always answers "no purchase" gets **84.5 % accuracy** but finds no buyer at all (precision = recall = F1 = 0). For this reason the main metric is the **F1-score of the purchase class**; accuracy, precision and recall are reported too.

---

## 2. Repository structure and how to run

The notebook was developed on **Google Colab with Google Drive**, and the paths in the code are currently hard-coded as follows:

```
Main_Folder (or the name you decided in cell 1.1)/
├── Datasets/
│   └── dataset_purchase (or the name you decided in cell 1.1)/
│       └── online_shoppers_intention.csv     <- INPUT: put the dataset here
└── Results/                                  <- OUTPUT: all figures (.png) are saved here
    ├── correlation_matrix.png
    ├── EDA/
    ├── Logistic_regression/
    ├── Decision_tree_classifier/
    ├── Random_forest/
    ├── Dense_neural_network/
    │   ├── Overfitted_model/
    │   ├── Light_model/
    │   └── Best_model/
    └── Conclusions/
```

**Steps**
1. Download `online_shoppers_intention.csv` from Kaggle and place it in `Datasets/dataset_purchase/`.
2. Create the `Results/` folder and its sub-folders (`savefig` does **not** create missing folders, so the notebook fails if they don't exist).
3. Open `Definitivo.ipynb` in Colab and run all cells from the top (the first cell mounts Drive).

To run it somewhere else (local / Kaggle) edit the two path variables only: the `pd.read_csv(...)` path in section 2 and the `folder` variable in section 1.2 (e.g. `folder = "results/"` plus `os.makedirs`). On Kaggle the dataset path is `/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv`.

**Requirements:** Python 3, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` (Keras). `keras-tuner` is needed only to repeat the grid search.

**Flags and seeds**
- `SEED = 42` for numpy, python, TensorFlow and all splits/models.
- `RUN_EXPENSIVE_CELLS = False` skips the Keras-Tuner grid search; the best hyper-parameters found are stored in the notebook. Set it to `True` (and `pip install keras-tuner`, uncomment the import) to repeat it.

---

## 3. How the analysis works

### 3.1 Exploratory analysis
- **Correlation with the target** (numeric columns only): `PageValues` is by far the strongest (+0.49); `ExitRates` (-0.21) and `BounceRates` (-0.15) are negative; page counts and durations are weakly positive (0.07-0.16); the others are about 0.
- **Class distribution**: 15.5 % positives, hence the choice of F1 and of threshold tuning.
- **Feature distributions**: counts and durations are strongly right-skewed.

### 3.2 Preprocessing
- **Feature selection.** Only the 9 numeric behavioural features are kept: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`.
  The other 8 were **removed because they were considered not useful for learning**: `OperatingSystems`, `Browser`, `Region`, `TrafficType` (anonymous integer codes with no meaningful order and about 0 correlation with the target), plus `Month`, `SpecialDay`, `VisitorType`, `Weekend`.
- **Split**: 80 % training / 20 % test, fixed seed (neural network section: an additional validation set).
- **Scaling**: `MinMaxScaler` to [0, 1], **fit on the training set only** to avoid leakage.
### 3.3 Models
| Model | Main settings |
|---|---|
| Baseline | Always predicts "no purchase" |
| Logistic Regression | `liblinear`, L2, `C=1` |
| Decision Tree | Gini, `max_depth=3` |
| Random Forest | 300 trees, `max_depth=3`, `min_samples_leaf=10`, `max_features='sqrt'` |
| Dense Neural Network | 3 architectures compared (large 3,265 params; light 233 params with L1/L2; best 593 params with dropout), Adam lr=1e-3, batch 32, 50 epochs, binary cross-entropy |

For the neural network the final architecture comes from a grid search over L1/L2 strength, dropout rates and size of the second layer: `l1 = l2 = 1e-5`, dropout 0.5 / 0.4, 8 units in the second layer.

### 3.4 Threshold tuning
A classifier outputs P(purchase) and predicts "purchase" if P >= t. t = 0.5 is only a convention. For each model the notebook scans t from 0 to 1 (step 0.01) and keeps the t with the highest F1. Lowering t increases recall and decreases precision; raising it does the opposite. Because only about 15 % of sessions are buyers, it is probable that the best threshold is not the classic standard threshold of 0.5, indeed this analysis goes deeper and tryis to find the best threshold, meaning the threhshold that maximize the F1-score.

### 3.5 Outputs
For each model: metrics, learning curves (or loss curves), the F1-vs-threshold curve and the confusion matrix. The final section compares all models in two tables/bar charts (standard vs best threshold). All figures are saved in `Results/` (see section 2).

---

## 4. Results (test set)

**Threshold 0.5**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | 0.845 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.864 | 0.775 | 0.260 | 0.390 |
| Decision Tree | 0.879 | 0.668 | 0.547 | 0.602 |
| Random Forest | 0.881 | 0.868 | 0.336 | 0.484 |
| Dense Neural Network | 0.880 | 0.756 | 0.441 | 0.557 |

**Best threshold (maximising F1)**

| Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.18 | 0.866 | 0.582 | 0.698 | 0.635 |
| Decision Tree | 0.14 | 0.872 | 0.586 | 0.793 | 0.674 |
| Random Forest | 0.31 | 0.878 | 0.603 | 0.774 | 0.678 |
| Dense Neural Network | 0.36 | 0.891 | 0.668 | 0.716 | 0.691 |

### What the results mean
- **Accuracy hides the real progress.** All models are only 2-5 points above the 84.5 % baseline, while F1 goes from 0 to about 0.7.
- **At threshold 0.5 the models are conservative.** Random Forest has the highest precision (0.87) but finds only a third of the buyers (recall 0.34). The Decision Tree has the best F1 at 0.5 simply because its probabilities are less conservative.
- **Threshold tuning matters more than the choice of model.** It raises F1 by +0.07 (Decision Tree) to +0.25 (Logistic Regression), trading precision for recall.
- **After tuning, models are close** (F1 0.635-0.691). The neural network is best, but its margin over Random Forest and Decision Tree (about 0.01-0.02) is small and likely within the noise of a single test split.
- **Feature importance (Random Forest).** `PageValues` accounts for about 68 % of the total importance, `ExitRates` about 12 %, `ProductRelated_Duration` about 8 %; the top three cover about 88 %, while the `Informational*` features are below 0.5 % each. This is consistent with the correlation analysis. `PageValues` is computed by Google Analytics from past transactions, so part of its predictive power is close to circular. The few dominant features also explain why very different models reach similar F1, and supports dropping the other features.
- **Learning curves.** Logistic Regression, Decision Tree and Random Forest show a small train/validation gap (low variance) and plateau early, i.e. they are limited by bias, not by lack of data. The large neural network shows mild overfitting; with dropout and regularisation the training loss is above the validation loss because dropout is active only during training.

---
