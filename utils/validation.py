"""
validation.py
--------------
Lightweight, dependency-free validation for the prediction form.
Returns a list of human-readable error strings; an empty list means
the input is valid.
"""

from utils.config import ALL_FIELDS


def validate_form(values: dict) -> list:
    errors = []

    for key, field in ALL_FIELDS.items():
        if key not in values:
            errors.append(f"Missing value for '{field['label']}'.")
            continue

        val = values[key]

        if field["type"] in ("int", "number"):
            try:
                num = float(val)
            except (TypeError, ValueError):
                errors.append(f"'{field['label']}' must be a number.")
                continue
            if num < field["min"] or num > field["max"]:
                errors.append(
                    f"'{field['label']}' must be between {field['min']} and {field['max']}."
                )

    # Cross-field logical checks
    age = values.get("Age")
    first_sex_age = values.get("First_Sexual_Intercourse_Age")
    if age is not None and first_sex_age is not None:
        try:
            if float(first_sex_age) > float(age):
                errors.append(
                    "Age at first sexual intercourse cannot be greater than current age."
                )
        except (TypeError, ValueError):
            pass

    smoking_years = values.get("Smoking_Years")
    if age is not None and smoking_years is not None:
        try:
            if float(smoking_years) > float(age):
                errors.append("Years smoking cannot exceed the patient's age.")
        except (TypeError, ValueError):
            pass

    return errors
