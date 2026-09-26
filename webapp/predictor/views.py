from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect

# Trained models live in the training/ folder next to webapp/
MODEL_DIR = Path(settings.BASE_DIR).parent / 'training'

# Load the models
knn = joblib.load(MODEL_DIR / 'knn_model.pkl')
log_reg = joblib.load(MODEL_DIR / 'log_reg_model.pkl')
rf = joblib.load(MODEL_DIR / 'rf_model.pkl')
scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
feature_names = joblib.load(MODEL_DIR / 'feature_names.pkl')

# The dataset stores categorical fields as numeric codes, so the text values
# from the form must be converted to the same codes the models were trained on.
CATEGORY_CODES = {
    'Sex': {'female': 0, 'male': 1},
    'Chest pain type': {'typical angina': 1, 'atypical angina': 2,
                        'non-anginal pain': 3, 'asymptomatic': 4},
    'FBS over 120': {'no': 0, 'yes': 1},
    'EKG results': {'normal': 0, 'having ST-T wave abnormality': 1,
                    'showing probable or definite left ventricular hypertrophy': 2},
    'Exercise angina': {'no': 0, 'yes': 1},
    'Slope of ST': {'upsloping': 1, 'flat': 2, 'downsloping': 3},
    'Thallium': {'normal': 3, 'fixed defect': 6, 'reversible defect': 7},
}


def preprocess_user_input(user_data):
    encoded = dict(user_data)
    for column, codes in CATEGORY_CODES.items():
        encoded[column] = codes[encoded[column]]
    df = pd.DataFrame([encoded])[list(feature_names)]
    return scaler.transform(df)


def index(request):
    if request.method == 'POST':
        user_data = {
            'Age': int(request.POST['age']),
            'Sex': request.POST['sex'],
            'Chest pain type': request.POST['chest_pain'],
            'BP': float(request.POST['bp']),
            'Cholesterol': float(request.POST['cholesterol']),
            'FBS over 120': request.POST['fbs'],
            'EKG results': request.POST['ekg'],
            'Max HR': float(request.POST['max_hr']),
            'Exercise angina': request.POST['exercise_angina'],
            'ST depression': float(request.POST['st_depression']),
            'Slope of ST': request.POST['slope_of_st'],
            'Number of vessels fluro': int(request.POST['num_vessels']),
            'Thallium': request.POST['thallium']
        }

        user_df = preprocess_user_input(user_data)

        knn_pred = int(knn.predict(user_df)[0])
        log_reg_pred = int(log_reg.predict(user_df)[0])
        rf_pred = int(rf.predict(user_df)[0])

        # Majority voting
        final_pred = (knn_pred + log_reg_pred + rf_pred) >= 2

        if final_pred:
            result = "The model predicts that the user has a chance of heart disease."
        else:
            result = "The model predicts that the user does not have a chance of heart disease."

        # Store the result in the session
        request.session['result'] = result
        return redirect('index')

    # Retrieve and clear the result from the session
    result = request.session.pop('result', '')

    return render(request, 'predictor/index.html', {'result': result})
