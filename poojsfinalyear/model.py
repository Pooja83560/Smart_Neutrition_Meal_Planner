# model_training.py
import os, json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
# --- config ---
BASE = os.path.dirname(__file__)
CSV = os.path.join(BASE, "diet_recommendations_dataset.csv")
OUT_DIR = os.path.join(BASE, "models")
os.makedirs(OUT_DIR, exist_ok=True)
# --- load data ---
df = pd.read_csv(CSV)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
# --- features & target (adjust if different) ---
num_cols = ['age','bmi','daily_caloric_intake','cholesterol_mg/dl','glucose_mg/dl']
cat_cols = ['gender','disease_type']
target = 'diet_recommendation'
# drop rows with missing target
df = df[df[target].notna()]
X = df[num_cols + cat_cols]
y = df[target]
# --- pipeline ---
num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
cat_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
preproc = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
pipe = ImbPipeline([('pre', preproc), ('smote', SMOTE(random_state=42)), ('clf', clf)])
# --- search ---
param_dist = {
    'clf__n_estimators': [50, 100],  # Reduced options
    'clf__max_depth': [6, 12],       # Reduced options
    'clf__min_samples_split': [2, 5] # Reduced options
}
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
search = RandomizedSearchCV(pipe, param_dist, n_iter=5, cv=cv, scoring='f1_macro', n_jobs=1, random_state=42, verbose=1)  # Reduced iterations, single job
search.fit(X_train, y_train)
best = search.best_estimator_
y_pred = best.predict(X_test)
# --- save & report ---
joblib.dump(best, os.path.join(OUT_DIR, 'best_model.joblib'))
report = classification_report(y_test, y_pred, output_dict=True)
with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(report, f, indent=2)
# --- save & report ---
joblib.dump(best, os.path.join(OUT_DIR, 'best_model.joblib'))
report = classification_report(y_test, y_pred, output_dict=True)
with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(report, f, indent=2)

print("Best params:", search.best_params_)
print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

# Print classification report with percentages
print("\nCLASSIFICATION REPORT:")
print("-" * 50)
print(classification_report(y_test, y_pred))

# Extract and display key metrics in percentage format
accuracy = report['accuracy'] * 100
macro_avg = report['macro avg']
weighted_avg = report['weighted avg']

print("\nKEY METRICS (in percentages):")
print("-" * 40)
print(f"Overall Accuracy:     {accuracy:.2f}%")
print(f"Macro Avg Precision:  {macro_avg['precision']*100:.2f}%")
print(f"Macro Avg Recall:     {macro_avg['recall']*100:.2f}%")
print(f"Macro Avg F1-Score:   {macro_avg['f1-score']*100:.2f}%")
print(f"Weighted Avg Precision: {weighted_avg['precision']*100:.2f}%")
print(f"Weighted Avg Recall:   {weighted_avg['recall']*100:.2f}%")
print(f"Weighted Avg F1-Score: {weighted_avg['f1-score']*100:.2f}%")

# Display confusion matrix with proper formatting
print("\nCONFUSION MATRIX:")
print("-" * 30)
cm = confusion_matrix(y_test, y_pred)
class_names = sorted(y.unique())

# Print header
print(f"{'Predicted':<12}", end="")
for name in class_names:
    print(f"{name:<12}", end="")
print("\n" + "-" * (12 + 12 * len(class_names)))

# Print matrix with row labels
for i, (row, true_label) in enumerate(zip(cm, class_names)):
    print(f"{true_label:<12}", end="")
    for val in row:
        print(f"{val:<12}", end="")
    print()

print("\n" + "="*60)