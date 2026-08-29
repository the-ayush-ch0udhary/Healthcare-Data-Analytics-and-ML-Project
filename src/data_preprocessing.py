"""
Data Preprocessing Pipeline for Hospital Patient Analytics
Merges raw patient data, diagnoses, outcomes, and clinical lab results
into a clean, standardized dataset for analysis and machine learning.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def preprocess_data(raw_dir: Path, output_file: Path) -> pd.DataFrame:
    """
    Ingests raw healthcare datasets, performs validation and transformations,
    and writes the cleaned dataset to the specified output file.
    """
    print("=" * 60)
    print("Starting Healthcare Data Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load raw files
    patients_path = raw_dir / "patients.csv"
    diagnoses_path = raw_dir / "diagnoses.csv"
    outcomes_path = raw_dir / "outcomes.csv"
    labs_path = raw_dir / "labs.csv"

    print(f"Loading raw datasets from: {raw_dir}")
    patients_df = pd.read_csv(patients_path)
    diagnoses_df = pd.read_csv(diagnoses_path)
    outcomes_df = pd.read_csv(outcomes_path)
    labs_df = pd.read_csv(labs_path)

    print(f"  Patients records : {len(patients_df)}")
    print(f"  Diagnoses records: {len(diagnoses_df)}")
    print(f"  Outcomes records : {len(outcomes_df)}")
    print(f"  Labs records     : {len(labs_df)}")

    # 2. Merge diagnoses and outcomes with patients
    merged_df = patients_df.merge(diagnoses_df, on="DiagnosisID", how="left")
    merged_df = merged_df.merge(outcomes_df, on="OutcomeID", how="left")

    # 3. Calculate Length of Stay if dates are present
    if "AdmissionDate" in merged_df.columns and "DischargeDate" in merged_df.columns:
        admission = pd.to_datetime(merged_df["AdmissionDate"], errors="coerce")
        discharge = pd.to_datetime(merged_df["DischargeDate"], errors="coerce")
        calculated_los = (discharge - admission).dt.days
        if "LengthOfStay" not in merged_df.columns:
            merged_df["LengthOfStay"] = calculated_los
        else:
            merged_df["LengthOfStay"] = merged_df["LengthOfStay"].fillna(calculated_los)

    # 4. Pivot Labs data by PatientID and TestName
    print("\nPivoting clinical laboratory measurements...")
    labs_pivot = labs_df.pivot_table(
        index="PatientID",
        columns="TestName",
        values="Result",
        aggfunc="first"
    ).reset_index()

    # 5. Merge pivoted labs into the main dataframe
    final_df = merged_df.merge(labs_pivot, on="PatientID", how="left")

    # 6. Data Validation and Cleaning
    print("\nValidating dataset...")
    expected_labs = [
        "Blood Pressure",
        "Blood Sugar",
        "Cholesterol",
        "Creatinine",
        "Hemoglobin",
        "Vitamin D"
    ]
    for col in expected_labs:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

    # Handle missing values if any
    null_counts = final_df.isnull().sum()
    if null_counts.sum() > 0:
        print("Warning: Missing values detected, imputing numeric with median:")
        print(null_counts[null_counts > 0])
        for col in expected_labs:
            if col in final_df.columns and final_df[col].isnull().any():
                final_df[col] = final_df[col].fillna(final_df[col].median())

    # 7. Reorder columns logically
    core_cols = [
        "PatientID", "Name", "Age", "Gender", "DiagnosisID",
        "AdmissionDate", "DischargeDate", "OutcomeID",
        "TreatmentCost", "LengthOfStay", "DiagnosisName", "OutcomeName"
    ]
    all_ordered = [c for c in core_cols if c in final_df.columns] + [
        c for c in expected_labs if c in final_df.columns
    ]
    final_df = final_df[all_ordered]

    # 8. Save cleaned dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"Cleaned dataset successfully saved to: {output_file}")
    print(f"Final shape: {final_df.shape}")
    print("=" * 60)

    return final_df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_FILE = BASE_DIR / "data" / "processed" / "healthcare_cleaned.csv"

    preprocess_data(RAW_DIR, PROCESSED_FILE)
