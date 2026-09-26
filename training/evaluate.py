"""Compare the three models with 5-fold cross-validation.

A single 80/20 split leaves only 54 test patients, so accuracy can swing a lot
depending on which patients land in the test set. Cross-validation averages
over five different splits and gives a more reliable estimate.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Heart_Disease_Prediction.csv').dropna()
x = df.drop('Heart Disease', axis=1)
y = (df['Heart Disease'] == 'Presence').astype(int)

models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(f"{'Model':<22}{'CV accuracy':>18}{'CV recall':>12}")
for name, model in models.items():
    # Scaler inside the pipeline so it is fit on training folds only
    scores = cross_validate(make_pipeline(StandardScaler(), model), x, y, cv=cv,
                            scoring=['accuracy', 'recall'])
    acc, rec = scores['test_accuracy'], scores['test_recall']
    print(f"{name:<22}{acc.mean():>11.1%} ± {acc.std():.1%}{rec.mean():>12.1%}")
