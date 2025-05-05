import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # contoh model berbasis pohon untuk Feature Importance
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from catboost import CatBoostClassifier

# Import SMOTE untuk oversampling
from imblearn.over_sampling import SMOTE

from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# ================== Load dan Prasiapkan Data ==================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop id
df.drop("id", axis=1, inplace=True)

# Ganti 'Unknown' jadi NaN
df.replace("Unknown", np.nan, inplace=True)

# Konversi kategorikal
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

# Imputasi nilai hilang
imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Fitur dan target
X = df_imputed.drop("stroke", axis=1)
y = df_imputed["stroke"]

print("Distribusi sebelum sampling:", Counter(y))

# ================== Split Data (Stratified) ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================== SMOTE Oversampling ==================
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# ================== Standarisasi ==================
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# ================== Inisialisasi Model ==================
logreg = LogisticRegression(class_weight='balanced', random_state=42)
rf     = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)  # Kurangi jumlah estimator
gb     = GradientBoostingClassifier(n_estimators=50, random_state=42)  # Kurangi jumlah estimator
cat    = CatBoostClassifier(verbose=0, random_state=42, scale_pos_weight=5)

# Voting classifier
voting = VotingClassifier(
    estimators=[('lr', logreg), ('rf', rf), ('gb', gb)],  # Hanya 3 model untuk mempercepat
    voting='soft'
)

# ================== Hyperparameter Tuning ==================
param_grid = {
    'lr__C': [0.01, 0.1, 1],  # Kurangi ruang pencarian
    'rf__n_estimators': [50, 100],  # Kurangi pilihan estimators
    'gb__n_estimators': [50, 100],  # Kurangi pilihan estimators
}

grid_search = GridSearchCV(
    estimator=voting,
    param_grid=param_grid,
    cv=2,  # Kurangi jumlah fold
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_resampled_scaled, y_resampled)
print("Best Parameters:", grid_search.best_params_)

# Model terbaik
best_voting = grid_search.best_estimator_

# ================== Prediksi dan Evaluasi ==================
y_proba = best_voting.predict_proba(X_test_scaled)[:, 1]
y_pred  = best_voting.predict(X_test_scaled)

print("\n=== Before Threshold Tuning ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
print("PR-AUC   :", auc(recall, precision))

# ================== Threshold Tuning ==================
best_threshold = 0.5
best_f1 = 0
for thresh in np.arange(0.0001, 0.5, 0.0005):
    y_thresh_pred = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_thresh_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print("\nBest threshold for F1:", round(best_threshold, 4))

y_final = (y_proba >= best_threshold).astype(int)
print("\n=== After Threshold Tuning ===")
print(classification_report(y_test, y_final, digits=4))
print("Accuracy :", accuracy_score(y_test, y_final))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))
# ================== Confusion Matrix ==================
cm = confusion_matrix(y_test, y_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# ================== ROC Curve ==================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()

# ================== Precision-Recall Curve ==================
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.savefig("precision_recall_curve.png")
plt.show()

# ================== Prediksi Distribusi ==================
plt.figure(figsize=(6, 5))
plt.hist(y_proba, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Predicted Probability of Stroke")
plt.ylabel("Frequency")
plt.savefig("prediction_distribution.png")
plt.show()

# ================== Performance Metrics Bar Plot ==================
accuracy = accuracy_score(y_test, y_final)
precision = precision_score(y_test, y_final)
recall = recall_score(y_test, y_final)
f1 = f1_score(y_test, y_final)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, scores, color=['lightgreen', 'skyblue', 'lightcoral', 'lightgoldenrodyellow'])
plt.title("Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.savefig("performance_metrics.png")
plt.show()

# ================== Learning Curve ==================
# Assuming you have stored the training and validation scores during training
# This is just a placeholder example, you should replace `train_scores` and `val_scores` with actual data
# train_scores, val_scores, epochs = ...

# plt.figure(figsize=(8, 5))
# plt.plot(epochs, train_scores, label="Training Score", color='blue')
# plt.plot(epochs, val_scores, label="Validation Score", color='orange')
# plt.title("Learning Curve")
# plt.xlabel("Epochs")
# plt.ylabel("Score")
# plt.legend(loc="lower right")
# plt.savefig("learning_curve.png")
# plt.show()

# ================== Feature Importance (For Tree-Based Models) ==================
# Example for a Random Forest model
# Assuming you are using a Random Forest Classifier
# If using another tree-based model, replace this with the appropriate model's feature importance method

model = RandomForestClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_
features = X_train.columns

sorted_idx = importances.argsort()
plt.figure(figsize=(10, 5))
plt.barh(features[sorted_idx], importances[sorted_idx], color='teal')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("feature_importance.png")
plt.show()

# ================== ROC Curve with Threshold Analysis ==================
thresholds_range = np.linspace(0, 1, 100)
tpr_threshold = []
fpr_threshold = []

for threshold in thresholds_range:
    y_pred_threshold = (y_proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_pred_threshold)
    tpr_threshold.append(tpr[1])  # True Positive Rate at threshold
    fpr_threshold.append(fpr[1])  # False Positive Rate at threshold

plt.figure(figsize=(8, 5))
plt.plot(fpr_threshold, tpr_threshold, color='green', lw=2)
plt.title("ROC Curve with Threshold Analysis")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("roc_with_threshold.png")
plt.show()
