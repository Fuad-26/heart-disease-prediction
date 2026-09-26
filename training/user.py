import joblib
import pandas as pd

# Load the models
knn = joblib.load('knn_model.pkl')
log_reg = joblib.load('log_reg_model.pkl')
rf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')  # Load feature names
le = joblib.load('label_encoder.pkl')  # Load label encoder

# The dataset stores categorical fields as numeric codes, so text answers
# must be converted to the same codes the models were trained on.
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


# Define the function to get user input
def get_user_input():
    user_data = {
        'Age': int(input("Enter Age: ")),
        'Sex': input("Enter Sex (male/female): "),
        'Chest pain type': input(
            "Enter Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic): "),
        'BP': float(input("Enter Blood Pressure: ")),
        'Cholesterol': float(input("Enter Cholesterol: ")),
        'FBS over 120': input("Enter FBS over 120 (yes/no): "),
        'EKG results': input(
            "Enter EKG results (normal, having ST-T wave abnormality, showing probable or definite left ventricular hypertrophy): "),
        'Max HR': float(input("Enter Maximum Heart Rate: ")),
        'Exercise angina': input("Enter Exercise induced angina (yes/no): "),
        'ST depression': float(input("Enter ST depression: ")),
        'Slope of ST': input("Enter Slope of ST (upsloping, flat, downsloping): "),
        'Number of vessels fluro': int(input("Enter Number of major vessels (0-3): ")),
        'Thallium': input("Enter Thallium stress test result (normal, fixed defect, reversible defect): ")
    }
    return user_data


# Define the function to preprocess user input
def preprocess_user_input(user_data):
    encoded = dict(user_data)
    for column, codes in CATEGORY_CODES.items():
        encoded[column] = codes[encoded[column].strip()]
    # Same column order as the training set
    df = pd.DataFrame([encoded])[list(feature_names)]
    df = scaler.transform(df)  # Scale the features
    return df


# Define the function to make predictions
def predict_heart_disease(user_data):
    user_df = preprocess_user_input(user_data)

    knn_pred = int(knn.predict(user_df)[0])
    log_reg_pred = int(log_reg.predict(user_df)[0])
    rf_pred = int(rf.predict(user_df)[0])

    # Optionally, you can combine the predictions using majority voting
    final_pred = (knn_pred + log_reg_pred + rf_pred) >= 2  # Majority voting

    return final_pred


# Main function to run the prediction
if __name__ == "__main__":
    user_data = get_user_input()
    prediction = predict_heart_disease(user_data)

    if prediction:
        print("The model predicts that the user has a chance of heart disease.")
    else:
        print("The model predicts that the user does not have a chance of heart disease.")
