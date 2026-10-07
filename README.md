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
