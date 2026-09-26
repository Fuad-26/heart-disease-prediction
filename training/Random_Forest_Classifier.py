import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
df = pd.read_csv('Heart_Disease_Prediction.csv')

# Drop missing values
df = df.dropna()

# Split the data into features and target variable
x = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Encode the target variable as integers
le = LabelEncoder()
y = le.fit_transform(y)

# Select specific features for training
features = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
            'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST',
            'Number of vessels fluro', 'Thallium']
x = x[features]

# Encode categorical variables
x = pd.get_dummies(x, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instantiate the scaler
scaler = StandardScaler()

# Scale the features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Save the model, scaler, and feature names
joblib.dump(rf, 'rf_model.pkl')

# Evaluate the model
rf_pred = rf.predict(X_test)
print("Random Forest Model")
print(f'Accuracy: {accuracy_score(y_test, rf_pred)}')
print(classification_report(y_test, rf_pred))
