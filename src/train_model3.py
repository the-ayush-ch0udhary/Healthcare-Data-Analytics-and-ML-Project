import pandas as pd
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "healthcare_cleaned.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print()


# ============================================================
# 3. CHECK DATA
# ============================================================

print("Columns:")
print(df.columns.tolist())
print()

print("Missing values:")
print(df.isnull().sum())
print()


# ============================================================
# 4. REMOVE IRRELEVANT / LEAKAGE COLUMNS
# ============================================================

columns_to_remove = [
    "PatientID",
    "Name",
    "AdmissionDate",
    "DischargeDate",
    "TreatmentCost",
    "LengthOfStay",
    "OutcomeID"
]

df_ml = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

print("=" * 60)
print("ML Dataset")
print("=" * 60)

print(f"ML dataset shape: {df_ml.shape}")
print()


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "Age",
    "Gender",
    "DiagnosisID",
    "Blood Pressure",
    "Blood Sugar",
    "Cholesterol",
    "Creatinine",
    "Hemoglobin",
    "Vitamin D"
]

target = "OutcomeName"

X = df_ml[features].copy()
y = df_ml[target].copy()


# ============================================================
# 6. CONVERT CATEGORICAL FEATURES
# ============================================================

X["Gender"] = X["Gender"].astype(str)
X["DiagnosisID"] = X["DiagnosisID"].astype(str)


categorical_features = [
    "Gender",
    "DiagnosisID"
]

numerical_features = [
    "Age",
    "Blood Pressure",
    "Blood Sugar",
    "Cholesterol",
    "Creatinine",
    "Hemoglobin",
    "Vitamin D"
]


print("Categorical features:")
print(categorical_features)

print()

print("Numerical features:")
print(numerical_features)

print()


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("=" * 60)
print("Target Distribution")
print("=" * 60)

print(
    y.value_counts()
)

print()

print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print()


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=" * 60)
print("Train / Test Split")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print()


# ============================================================
# 9. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 11. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            rf_model
        )
    ]
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("=" * 60)
print("Training Random Forest")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")
print()


# ============================================================
# 13. PREDICTIONS
# ============================================================

predictions = pipeline.predict(
    X_test
)


# ============================================================
# 14. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy Percentage: {accuracy * 100:.2f}%"
)

print()


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    predictions
)

print("Classification Report:")
print(report)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=pipeline.classes_
)

print("Confusion Matrix:")
print(cm)
print()


# ============================================================
# 17. SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=pipeline.classes_,
    yticklabels=pipeline.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")

plt.tight_layout()

confusion_matrix_path = (
    MODEL_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    MODEL_DIR
    / "classification_report.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "Random Forest Classification Report\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Accuracy Percentage: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        report
    )


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

classifier = pipeline.named_steps[
    "classifier"
]

preprocessor_fitted = pipeline.named_steps[
    "preprocessor"
]

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)

importance = classifier.feature_importances_

feature_importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)

feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("=" * 60)
print("Top Feature Importances")
print("=" * 60)

print(
    feature_importance_df.head(15)
)

print()


# ============================================================
# 20. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_df.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
)


# ============================================================
# 21. FEATURE IMPORTANCE PLOT
# ============================================================

top_features = (
    feature_importance_df
    .head(15)
    .sort_values(
        by="Importance"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. SAVE COMPLETE PIPELINE
# ============================================================

model_path = (
    MODEL_DIR
    / "best_model.pkl"
)

joblib.dump(
    pipeline,
    model_path
)

print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Model: {model_path}"
)

print(
    f"Confusion Matrix: {confusion_matrix_path}"
)

print(
    f"Classification Report: {report_path}"
)

print(
    f"Feature Importance: "
    f"{MODEL_DIR / 'feature_importance.csv'}"
)

print()
print("Training completed successfully.")