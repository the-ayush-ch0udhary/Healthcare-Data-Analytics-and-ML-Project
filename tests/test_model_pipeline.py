"""
Unit tests for Machine Learning Model Pipeline and Inference.
"""

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = self.base_dir / "models" / "best_model.pkl"
        self.metadata_path = self.base_dir / "models" / "model_metadata.json"

    def test_model_and_metadata_exist(self):
        self.assertTrue(self.model_path.exists(), "Trained model best_model.pkl is missing.")
        self.assertTrue(self.metadata_path.exists(), "Model metadata JSON is missing.")

    def test_metadata_structure(self):
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.assertIn("accuracy", metadata)
        self.assertGreaterEqual(metadata["accuracy"], 0.75, "Model accuracy should be at least 75%.")
        self.assertIn("classes", metadata)
        self.assertEqual(len(metadata["classes"]), 3)
        self.assertIn("class_metrics", metadata)

    def test_single_prediction_inference(self):
        model = joblib.load(self.model_path)
        sample_input = pd.DataFrame(
            {
                "Age": [45],
                "Gender": ["M"],
                "DiagnosisID": ["1"],
                "Blood Pressure": [120.0],
                "Blood Sugar": [95.0],
                "Cholesterol": [180.0],
                "Creatinine": [0.9],
                "Hemoglobin": [14.5],
                "Vitamin D": [35.0],
            }
        )

        prediction = model.predict(sample_input)
        probabilities = model.predict_proba(sample_input)

        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], ["Recovered", "Complicated", "Deceased"])
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(np.sum(probabilities[0]), 1.0, places=4)

    def test_batch_prediction_inference(self):
        model = joblib.load(self.model_path)
        batch_input = pd.DataFrame(
            {
                "Age": [30, 75, 50],
                "Gender": ["F", "M", "F"],
                "DiagnosisID": ["2", "3", "9"],
                "Blood Pressure": [115.0, 160.0, 130.0],
                "Blood Sugar": [90.0, 180.0, 110.0],
                "Cholesterol": [170.0, 240.0, 195.0],
                "Creatinine": [0.8, 2.2, 1.1],
                "Hemoglobin": [13.5, 10.2, 14.0],
                "Vitamin D": [40.0, 18.0, 30.0],
            }
        )

        batch_preds = model.predict(batch_input)
        self.assertEqual(len(batch_preds), 3)


if __name__ == "__main__":
    unittest.main()
