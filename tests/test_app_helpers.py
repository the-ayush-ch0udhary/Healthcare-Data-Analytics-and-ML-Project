"""
Unit tests for App Helpers and Clinical Range Evaluators.
"""

import sys
import unittest
from pathlib import Path

# Add streamlit_app to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "streamlit_app"))

from utils.theme import get_clinical_status, get_model_metadata


class TestAppHelpers(unittest.TestCase):
    def test_clinical_status_blood_pressure(self):
        stat, badge, desc = get_clinical_status("Blood Pressure", 115.0)
        self.assertEqual(stat, "Normal")
        self.assertEqual(badge, "badge-normal")

        stat, badge, desc = get_clinical_status("Blood Pressure", 155.0)
        self.assertIn("Hypertension", stat)
        self.assertEqual(badge, "badge-danger")

    def test_clinical_status_blood_sugar(self):
        stat, badge, desc = get_clinical_status("Blood Sugar", 85.0)
        self.assertEqual(stat, "Normal Fasting")
        self.assertEqual(badge, "badge-normal")

        stat, badge, desc = get_clinical_status("Blood Sugar", 145.0)
        self.assertEqual(stat, "Diabetic Range")
        self.assertEqual(badge, "badge-danger")

    def test_model_metadata_loader(self):
        meta = get_model_metadata()
        self.assertIsInstance(meta, dict)
        self.assertIn("model_name", meta)
        self.assertIn("accuracy", meta)


if __name__ == "__main__":
    unittest.main()
