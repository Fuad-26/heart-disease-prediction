# Heart Disease Prediction

Predicts whether a patient is likely to have heart disease from 13 clinical measurements, comparing three machine learning models and serving the prediction through a Django web form.

## Dataset

270 patient records (150 without heart disease, 120 with) from the UCI Statlog Heart dataset. Each record has 13 features, including age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, ST depression, number of major vessels seen on fluoroscopy, and thallium stress test result.

## Approach

1. Loaded and cleaned the data with pandas and encoded the target (Presence / Absence).
2. Scaled the features with `StandardScaler` and split the data 80/20 for training and testing.
3. Trained K-Nearest Neighbors, Logistic Regression and Random Forest classifiers with scikit-learn.
4. Evaluated each model on the hold-out set and with 5-fold cross-validation.
5. Saved the models with joblib and built a Django web app where a user enters the 13 values and the three models vote on the prediction.

## Results

| Model | Hold-out accuracy (54 patients) | 5-fold CV accuracy | 5-fold CV recall |
|---|---|---|---|
| Logistic Regression | 90.7% | 83.3% | 79.2% |
| KNN (k=5) | 81.5% | 81.1% | 75.8% |
| Random Forest | 75.9% | 83.0% | 76.7% |

Recall is the share of patients with heart disease that the model correctly identifies, which matters more than accuracy in a medical setting because a missed case is the costly mistake.

The hold-out set is small, so cross-validation gives the more reliable estimate: roughly 81–83% accuracy for all three models. Logistic Regression is the best choice here since it matches the others while being the simplest and easiest to interpret.

**Key findings:** the strongest predictors were the number of major vessels seen on fluoroscopy, chest pain type, thallium stress test result and ST depression. Patients with heart disease reached a lower maximum heart rate on average (about 139 vs 158 bpm).

## Project structure

```
training/
  Heart_Disease_Prediction.csv   dataset
  train_models.py                trains and saves all three models
  KNN.py                         trains and evaluates KNN
  Logistic_Regression.py         trains and evaluates Logistic Regression
  Random_Forest_Classifier.py    trains and evaluates Random Forest
  evaluate.py                    5-fold cross-validation comparison
  user.py                        command-line prediction
  *.pkl                          saved models, scaler and feature names
webapp/
  manage.py
  Application/                   Django project settings
  predictor/                     prediction view and HTML form
```

## How to run

Requires Python 3.12.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# Evaluate the models
cd training
python evaluate.py

# Run the web app
cd ../webapp
python manage.py migrate
python manage.py runserver
```

Then open http://127.0.0.1:8000 in a browser.

## Tools

Python, pandas, NumPy, scikit-learn, joblib, Django, HTML/CSS

## Limitations

This is a learning project trained on 270 patients and is not a medical tool. The web app runs with Django's development settings and is meant for local demos only.
