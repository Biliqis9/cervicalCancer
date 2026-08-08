"""
preprocessing.py
-----------------
Reproduces the notebook's preprocessing pipeline for a single patient
record submitted through the form:

  1. Assemble raw fields into a DataFrame in the imputer's training order.
  2. Apply the fitted IterativeImputer (MICE) -- imputer.pkl.
  3. Engineer Smoking_Index (see ASSUMPTION note in utils/config.py).
  4. Reorder columns to exactly match feature_names.pkl / the model's
     training order before prediction.

Binary Yes/No fields are encoded as 1/0 before imputation, matching
typical scikit-learn numeric-only pipelines.
"""

import pandas as pd

BINARY_YES_NO_FIELDS = ["STD_HPV", "Dx_Cancer", "Dx_CIN", "Dx_HPV"]


def encode_binary(value: str) -> int:
    return 1 if str(value).strip().lower() == "yes" else 0


def compute_smoking_index(smoking_years: float, packs_per_year: float) -> float:
    """
    ASSUMPTION (verify against your training notebook):
    Smoking_Index = Smoking_Years * Packs_Per_Year

    This is the only feature the model uses that the imputer does not
    know about, which means it must be engineered *after* imputation.
    If your notebook computes it differently (e.g. a weighted formula,
    a binned/categorical index, or something involving Age), update
    this single function -- nothing else in the app needs to change.
    """
    return float(smoking_years) * float(packs_per_year)


def build_raw_dataframe(form_values: dict, imputer_feature_order: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame from form values, encoded numerically
    and ordered exactly as the imputer expects (imputer.feature_names_in_).
    """
    row = {}
    for col in imputer_feature_order:
        val = form_values[col]
        if col in BINARY_YES_NO_FIELDS:
            row[col] = encode_binary(val)
        else:
            row[col] = float(val)
    return pd.DataFrame([row], columns=imputer_feature_order)


def run_preprocessing_pipeline(
    form_values: dict,
    imputer,
    model_feature_order: list,
) -> pd.DataFrame:
    """
    Full pipeline: raw -> impute -> engineer Smoking_Index -> reorder
    to match the model's training column order. Returns a one-row
    DataFrame ready for model.predict_proba().
    """
    imputer_cols = list(imputer.feature_names_in_)

    raw_df = build_raw_dataframe(form_values, imputer_cols)

    imputed_array = imputer.transform(raw_df)
    imputed_df = pd.DataFrame(imputed_array, columns=imputer_cols)

    if "Smoking_Index" in model_feature_order:
        imputed_df["Smoking_Index"] = compute_smoking_index(
            imputed_df["Smoking_Years"].iloc[0],
            imputed_df["Packs_Per_Year"].iloc[0],
        )

    missing = [c for c in model_feature_order if c not in imputed_df.columns]
    if missing:
        raise ValueError(
            f"Preprocessing produced a frame missing required model columns: {missing}. "
            "This usually means feature_names.pkl doesn't match the imputer's "
            "expected inputs plus engineered features -- check utils/config.py."
        )

    return imputed_df[model_feature_order]
