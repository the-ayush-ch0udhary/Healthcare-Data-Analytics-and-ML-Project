"""
Unit tests for Healthcare Data Pipeline & Preprocessing.
"""

import unittest
from pathlib import Path
import pandas as pd


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.cleaned_data_path = self.base_dir / "data" / "processed" / "healthcare_cleaned.csv"

    def test_cleaned_dataset_exists(self):
        self.assertTrue(
            self.cleaned_data_path.exists(),
            f"Cleaned dataset missing at: {self.cleaned_data_path}",
        )

    def test_dataset_shape_and_columns(self):
        df = pd.read_csv(self.cleaned_data_path)
        self.assertGreater(len(df), 1000, "Dataset should contain at least 1000 records.")

        expected_columns = [
            "PatientID",
            "Name",
            "Age",
            "Gender",
            "DiagnosisID",
            "OutcomeID",
            "TreatmentCost",
            "LengthOfStay",
            "DiagnosisName",
            "OutcomeName",
            "Blood Pressure",
            "Blood Sugar",
            "Cholesterol",
            "Creatinine",
            "Hemoglobin",
            "Vitamin D",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing expected column: {col}")

    def test_no_critical_nulls(self):
        df = pd.read_csv(self.cleaned_data_path)
        critical_cols = [
            "PatientID",
            "Age",
            "Gender",
            "DiagnosisID",
            "OutcomeName",
            "Blood Pressure",
            "Blood Sugar",
            "Cholesterol",
            "Creatinine",
            "Hemoglobin",
            "Vitamin D",
        ]
        null_counts = df[critical_cols].isnull().sum()
        self.assertEqual(
            null_counts.sum(),
            0,
            f"Found unexpected missing values in critical columns: {null_counts[null_counts > 0]}",
        )

    def test_target_classes(self):
        df = pd.read_csv(self.cleaned_data_path)
        unique_outcomes = set(df["OutcomeName"].dropna().unique())
        expected_outcomes = {"Recovered", "Complicated", "Deceased"}
        self.assertEqual(unique_outcomes, expected_outcomes)


if __name__ == "__main__":
    unittest.main()
