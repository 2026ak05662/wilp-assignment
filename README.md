# Student Outcome Early-Warning: ML Assignment 2

Live app: https://student-dropout-warning.streamlit.app/
Repository: https://github.com/2026ak05662/wilp-assignment

## a. Problem statement

Many university students leave before completing their degree, and support is most useful when it starts early. The goal of this project is to predict, for each student, whether they will **drop out**, **stay enrolled** beyond the normal course duration, or **graduate**, using only information available at enrollment and after the **first semester**. This is framed as a 3-class classification problem, and five classifiers are trained and compared on the same data split.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/697 |
| Origin | Student records from a higher education institution in Portugal |
| Instances | 4,424 students |
| Features | 36 in the original data; 30 used after preprocessing (see below) |
| Target | `Target`: Dropout / Enrolled / Graduate (multi-class) |
| Class balance | Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%) |

The features cover demographics (age, gender, nationality, marital status), socio-economic background (parents' qualifications and occupations), application details (mode, course, admission grade), financial status (debtor, tuition fees up to date, scholarship), 1st-semester academic results, and regional economic indicators (unemployment, inflation, GDP).

**Preprocessing choices**

- **Early-warning setup:** the six "Curricular units 2nd sem" columns were removed, so the models only use what is known by the end of the first semester. This leaves 30 features.
- **Category codes:** columns such as `Course`, `Application mode`, `Nacionality` and the parents' qualifications and occupations are stored as numbers but are really category codes, so they are one-hot encoded instead of scaled. An experiment comparing both options is described in section d.
- **Numeric columns** are standardised (mean 0, standard deviation 1), which matters for KNN, SVM and Logistic Regression.
- **Missing values:** the dataset has none, but median / most-frequent imputation is included in the pipeline so the app does not fail on an incomplete upload.
- **Column names:** a stray tab character in "Daytime/evening attendance" was stripped.
- **Split:** stratified 80/20 train/test split (`random_state=42`), giving 3,539 training and 885 test students. The test split is saved as `test_data.csv`.

The scaler and encoder are saved inside each model's pipeline, so the app applies exactly the same preprocessing to uploaded data without retraining anything.

## c. GitHub repository

https://github.com/2026ak05662/wilp-assignment

```
wilp-assignment/
├── app.py                 Streamlit app (loads saved models, evaluates uploaded test CSV)
├── train_models.py        Data preparation, tuning and training of the five models
├── explore_data.ipynb     Exploratory data analysis
├── requirements.txt
├── README.md
├── test_data.csv          Held-out test split (885 rows, features + Target)
├── data/data.csv          Original UCI dataset
├── model/
│   ├── logistic_regression.joblib, decision_tree.joblib, knn.joblib,
│   │   naive_bayes.joblib, support_vector_machine.joblib   (full pipelines)
│   ├── label_encoder.joblib, metadata.json
│   ├── metrics_train_run.csv              final results (one-hot encoding)
│   └── metrics_codes_as_numbers.csv       experiment results (codes as numbers)
└── .streamlit/config.toml  App theme
```

Run locally:
```
pip install -r requirements.txt
python train_models.py        # retrains and saves the models
streamlit run app.py
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 885-student test split. Precision, recall and F1 use weighted averaging across the three classes; AUC is one-vs-rest, weighted.

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=8` |
| KNN | `n_neighbors=7` |
| Naive Bayes | Gaussian |
| Support Vector Machine | RBF kernel, `probability=True` |

**Hyperparameter tuning.** To select the final hyperparameters, I tested different values using 5-fold cross-validation on the training set only and compared their weighted F1 scores. For the Decision Tree, I tried `max_depth` values of 4, 6, 8, 10, 12 and None. A depth of 8 gave the best score (0.6947), so I kept it for the final model; the unlimited tree scored noticeably lower (0.6568), which suggests it overfits the training data. For KNN, I tested `n_neighbors` values of 3, 5, 7, 9, 11, 15 and 21. The best score was with 9 neighbours (0.6632), but 7 neighbours was almost identical (0.6615), a difference well within normal cross-validation variation, so I kept `n_neighbors=7` rather than retraining for a negligible gain.

| Decision Tree `max_depth` | 4 | 6 | **8** | 10 | 12 | None |
|---|---|---|---|---|---|---|
| CV weighted F1 | 0.6777 | 0.6939 | **0.6947** | 0.6868 | 0.6839 | 0.6568 |

| KNN `n_neighbors` | 3 | 5 | **7** | 9 | 11 | 15 | 21 |
|---|---|---|---|---|---|---|---|
| CV weighted F1 | 0.6390 | 0.6572 | **0.6615** | 0.6632 | 0.6628 | 0.6550 | 0.6495 |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7492 | 0.8769 | 0.7327 | 0.7492 | 0.7362 | 0.5827 |
| Decision Tree | 0.7028 | 0.8132 | 0.6895 | 0.7028 | 0.6918 | 0.5056 |
| KNN | 0.6893 | 0.8063 | 0.6726 | 0.6893 | 0.6749 | 0.4801 |
| Naive Bayes | 0.2441 | 0.6752 | 0.5531 | 0.2441 | 0.1675 | 0.1242 |
| Support Vector Machine | 0.7299 | 0.8845 | 0.7098 | 0.7299 | 0.7092 | 0.5489 |

### Encoding experiment

The models were also trained with the category-code columns treated as plain numbers instead of one-hot encoded. Everything else was kept the same.

| Model | Accuracy (one-hot) | Accuracy (codes as numbers) | MCC (one-hot) | MCC (codes as numbers) |
|---|---|---|---|---|
| Logistic Regression | 0.7492 | 0.7345 | 0.5827 | 0.5557 |
| Decision Tree | 0.7028 | 0.7017 | 0.5056 | 0.5010 |
| KNN | 0.6893 | 0.6463 | 0.4801 | 0.4073 |
| Naive Bayes | 0.2441 | 0.6226 | 0.1242 | 0.3608 |
| Support Vector Machine | 0.7299 | 0.7141 | 0.5489 | 0.5206 |

One-hot encoding was kept for the final models because it is the correct treatment for nominal codes and gives the best overall results.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression gives the strongest overall results, with the highest accuracy (74.92%), F1-score (0.7362) and MCC (0.5827). It also identifies 214 of the 284 students who dropped out, which is about 75%, with 55 students incorrectly flagged as dropouts. However, it has difficulty identifying students in the Enrolled category: it correctly identifies only 53 of 159 students in this group, and many are predicted as graduates. When category codes are treated as ordinary numbers instead of being one-hot encoded, its accuracy falls slightly to 73.45%. |
| Decision Tree | The Decision Tree achieves 70.28% accuracy and an MCC of 0.5056. It correctly catches 182 of the 284 dropouts (about 64%), while incorrectly flagging 63 other students as dropouts. Its accuracy changes very little when category codes are treated as numbers (70.28% to 70.17%). This suggests that, for this particular tree and dataset, the resulting splits are largely similar under the two encoding approaches. Its AUC of 0.8132 is third among the five models, clearly below Logistic Regression and SVM, indicating weaker class separation. |
| KNN | KNN achieves 68.93% accuracy and an MCC of 0.4801. It identifies 185 of the 284 dropouts (about 65%). Its performance falls more noticeably when category codes are treated as numbers: accuracy drops from 68.93% to 64.63%, and MCC from 0.4801 to 0.4073. This affects a distance-based model because assigning numbers to categories can make unrelated categories appear closer or farther apart than they really are. One-hot encoding avoids implying this artificial order. |
| Naive Bayes | Naive Bayes performs poorly with the final one-hot encoded features, achieving only 24.41% accuracy and an MCC of 0.1242. It predicts just 99 students as dropouts and catches only 57 of the 284 actual dropouts (about 20%). Its accuracy improves to 62.26% when category codes are kept as numbers, but that does not make numeric coding the more appropriate choice for nominal categories. The weak result with one-hot encoding may be related to the many sparse, low-frequency category columns and the correlations between some first-semester academic features. These characteristics can make the assumptions of Gaussian Naive Bayes a poor fit for this dataset. |
| Support Vector Machine | SVM has the highest AUC (0.8845), meaning it is strongest at ranking students by the model's estimated class probabilities. It achieves 72.99% accuracy and an MCC of 0.5489, placing it second to Logistic Regression on these measures. It identifies 194 of the 284 dropouts (about 68%) and produces 51 false alarms. In other words, it separates the classes well overall, but its final class predictions are slightly less accurate than Logistic Regression's. Treating category codes as numbers also reduces its accuracy, from 72.99% to 71.41%. |
| **Overall winner for this dataset** | Logistic Regression is the best overall choice for this dataset because it has the highest accuracy, F1-score and MCC, and it catches the largest number of students who actually drop out (214 of 284). SVM is a close alternative if ranking performance is the main priority, as it has the highest AUC and slightly fewer false alarms. None of the models handles the Enrolled group particularly well, so predictions for that group should be treated with caution. The MCC scores are also lower than the accuracy scores across the models, reminding us that accuracy alone does not tell the full story of performance across all three classes. |

## Streamlit app features

- Upload of the test CSV, with a button to download the sample `test_data.csv`
- Model selection dropdown for all five classifiers
- Six evaluation metrics for the selected model
- Confusion matrix and classification report
- Side-by-side comparison of all five models, with a bar chart by metric
- Early-warning panel: dropouts caught, missed and false alarms for each model
- Encoding experiment results

The app only loads the saved models and predicts; it does not retrain on uploaded data.

## Note

An AI assistant was used for guidance during this assignment.
