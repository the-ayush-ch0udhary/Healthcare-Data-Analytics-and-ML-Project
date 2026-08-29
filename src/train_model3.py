"""
Machine Learning Training and Model Benchmarking Pipeline
Hospital Patient Analytics

This script:
1. Loads preprocessed healthcare dataset.
2. Benchmarks multiple classification algorithms (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree).
3. Selects the champion model and saves the trained Pipeline (best_model.pkl).
4. Generates model_metadata.json for dynamic UI integration.
5. Saves classification report, confusion matrix, feature importances, and model comparison CSV.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def run_training_pipeline():
    print("=" * 60)
    print("Hospital Patient Analytics - Model Training & Benchmarking")
    print("=" * 60)

    # 1. Project Paths
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "healthcare_cleaned.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Dataset
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # 3. Define Features and Target
    categorical_features = ["Gender", "DiagnosisID"]
    numerical_features = [
        "Age",
        "Blood Pressure",
        "Blood Sugar",
        "Cholesterol",
        "Creatinine",
        "Hemoglobin",
        "Vitamin D",
    ]
    features = categorical_features + numerical_features
    target = "OutcomeName"

    X = df[features].copy()
    y = df[target].copy()

    # Convert categoricals to string for robust encoding
    for col in categorical_features:
        X[col] = X[col].astype(str)

    print(f"Features ({len(features)}): {features}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).mul(100).round(2)}")

    # 4. Train / Test Split (Stratified 80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 5. Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # 6. Benchmark Multiple Models
    print("\n" + "=" * 60)
    print("Benchmarking Multiple Classification Models")
    print("=" * 60)

    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
    }

    benchmark_results = []

    for name, clf in candidate_models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)

        benchmark_results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Accuracy_Pct": f"{acc * 100:.2f}%",
                "Precision_Macro": round(prec_macro, 4),
                "Recall_Macro": round(rec_macro, 4),
                "F1_Macro": round(f1_macro, 4),
                "F1_Weighted": round(f1_weighted, 4),
                "Training_Time_Sec": train_time,
            }
        )
        print(f"  {name:22s} | Accuracy: {acc*100:.2f}% | F1-Macro: {f1_macro:.4f} | Time: {train_time}s")

    benchmark_df = pd.DataFrame(benchmark_results).sort_values(by="Accuracy", ascending=False)
    comparison_csv_path = model_dir / "model_comparison.csv"
    benchmark_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison saved to: {comparison_csv_path}")

    # 7. Champion Model - Random Forest Full Pipeline
    print("\n" + "=" * 60)
    print("Training Final Champion Pipeline (Random Forest)")
    print("=" * 60)

    best_clf = candidate_models["Random Forest"]
    champion_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_clf),
        ]
    )
    champion_pipeline.fit(X_train, y_train)
    predictions = champion_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 8. Classification Report & Confusion Matrix
    class_labels = list(champion_pipeline.classes_)
    report_dict = classification_report(
        y_test, predictions, target_names=class_labels, output_dict=True
    )
    report_text = classification_report(y_test, predictions, target_names=class_labels)
    cm = confusion_matrix(y_test, predictions, labels=class_labels)

    print(f"Champion Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:\n", report_text)

    # 9. Feature Importance
    preprocessor_fitted = champion_pipeline.named_steps["preprocessor"]
    classifier_fitted = champion_pipeline.named_steps["classifier"]
    feature_names = list(preprocessor_fitted.get_feature_names_out())
    importances = classifier_fitted.feature_importances_

    feature_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    feature_imp_df.to_csv(model_dir / "feature_importance.csv", index=False)

    # 10. Generate High-Res Visualizations
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted Outcome", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Outcome", fontsize=11, fontweight="bold")
    plt.title("Random Forest - Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_feat = feature_imp_df.head(15).sort_values(by="Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top_feat["Feature"], top_feat["Importance"], color="#3b82f6")
    plt.xlabel("Importance Score", fontsize=11, fontweight="bold")
    plt.title("Top Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(model_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 11. Save Classification Report Text
    with open(model_dir / "classification_report.txt", "w") as f:
        f.write("Random Forest Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Accuracy Percentage: {accuracy * 100:.2f}%\n\n")
        f.write(report_text)

    # 12. Build and Save Detailed Model Metadata JSON
    metadata = {
        "model_name": "Random Forest",
        "accuracy": round(float(accuracy), 4),
        "accuracy_pct": f"{accuracy * 100:.2f}%",
        "dataset_total": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classes": class_labels,
        "class_metrics": {
            cls: {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall": round(float(report_dict[cls]["recall"]), 4),
                "f1_score": round(float(report_dict[cls]["f1-score"]), 4),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in class_labels
        },
        "macro_avg": {
            "precision": round(float(report_dict["macro avg"]["precision"]), 4),
            "recall": round(float(report_dict["macro avg"]["recall"]), 4),
            "f1_score": round(float(report_dict["macro avg"]["f1-score"]), 4),
        },
        "weighted_avg": {
            "precision": round(float(report_dict["weighted avg"]["precision"]), 4),
            "recall": round(float(report_dict["weighted avg"]["recall"]), 4),
            "f1_score": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        },
        "confusion_matrix": {
            "labels": class_labels,
            "matrix": cm.tolist(),
        },
        "top_features": feature_imp_df.head(15).to_dict(orient="records"),
        "trained_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "n_estimators": 300,
            "random_state": 42,
            "test_split": 0.20,
            "features_count": len(features),
        },
    }

    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 13. Save Champion Model Pipeline
    model_path = model_dir / "best_model.pkl"
    joblib.dump(champion_pipeline, model_path)

    print("\n" + "=" * 60)
    print("Training Pipeline Successfully Completed!")
    print(f"  Model file      : {model_path}")
    print(f"  Metadata JSON   : {model_dir / 'model_metadata.json'}")
    print(f"  Comparison CSV  : {comparison_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()