# Customer Churn Prediction – Machine Learning Lab Mini Project

Ramdeobaba University, Nagpur · B.Tech CSE · Machine Learning Lab

An end-to-end project that ties together every experiment of the lab: EDA, preprocessing,
classical classifiers, ensembles, hyper-parameter tuning, clustering and a Streamlit web UI.

## Run it
```bash
pip install -r requirements.txt
python generate_data.py      # creates data/churn.csv (messy, 3060 rows)
python train.py              # cleans, splits 80:20, tunes 7 models -> artifacts/
streamlit run app.py         # web UI (you can also press "Train" inside the UI)
```
Want your own dataset (e.g. from Kaggle)? Upload the CSV in the sidebar, choose the target column and ID columns, press **Train**.

## How it maps to the syllabus
| Lab experiment | Where in the project |
|---|---|
| 1 EDA & preprocessing (head/tail/describe/shape, nulls, duplicates, outliers, encoding, uni/bi/multivariate, scaling, 80:20 split) | Tab 1 + `ml_pipeline.py` (`basic_clean`, `IQRCapper`, `build_preprocessor`) |
| 4 k-NN | `KNN` model |
| 5 Decision Tree + metrics + confusion matrix + tuning | `Decision Tree`, Tab 3/4 |
| 6 SVM, kernels and value of C | `SVM` grid over `kernel` ∈ {linear, rbf, poly}, `C` ∈ {0.1, 1, 10} |
| 7 k-Means / Agglomerative clustering | Tab 6 (elbow, silhouette, PCA view, cluster profile) |
| 8 Random Forest + ensemble parameters | `Random Forest`, `AdaBoost` grids (n_estimators, depth, learning rate) |
| 11 Mini project: ≥4 tuned models, no leakage, web UI | the whole app |

## Data-leakage prevention
1. Duplicates are removed **before** the split.
2. Stratified 80:20 split happens **before** any statistic is learned.
3. Imputer → IQR outlier capping → One-Hot → StandardScaler live inside a `Pipeline`, so they are fitted on training folds only (also inside each GridSearchCV fold).

## Models (all tuned with 5-fold stratified GridSearchCV, scoring = F1)
Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, AdaBoost.
Evaluated on the untouched 20% test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves.

## Sample results (synthetic data)
Naive Bayes had the best test F1 (≈0.63, recall 0.72); AdaBoost and Logistic Regression followed. Results will differ on a real dataset.

## Viva talking points
* Why F1 for tuning? Churn is imbalanced, accuracy alone hides missed churners.
* Why median imputation and IQR capping? They are robust to the outliers themselves.
* Why scale before KNN/SVM/LR but not trees? Distance/margin-based models are scale-sensitive.
* Larger `C` in SVM → narrower margin, lower bias, higher overfitting risk.
* Random Forest: more trees reduce variance; deeper trees raise variance.

## Files
`generate_data.py` dataset · `ml_pipeline.py` pipeline & models · `train.py` CLI training · `app.py` Streamlit UI
