"""
config.py
---------
Central configuration: paths, colors, feature metadata, and the field
definitions used to build the input form. Keeping this in one place
means the form, validation, and preprocessing all stay in sync.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "model.pkl"
IMPUTER_PATH = MODELS_DIR / "imputer.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.pkl"

# Optional artifacts (Model Performance / SHAP pages degrade gracefully
# if these are not present -- see README.md "Completing the app").
X_TEST_PATH = MODELS_DIR / "X_test.pkl"
Y_TEST_PATH = MODELS_DIR / "y_test.pkl"
Y_PROB_TEST_PATH = MODELS_DIR / "y_prob_test.pkl"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"

# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------
COLORS = {
    "dark_blue": "#0B2545",
    "teal": "#0F9B8E",
    "soft_green": "#4CAF7D",
    "white": "#FFFFFF",
    "bg": "#F4F8FB",
    "card_bg": "#FFFFFF",
    "text": "#1B2430",
    "muted": "#5B6B79",
    "danger": "#D64545",
    "warning": "#E3A92B",
    "success": "#2E9E5B",
    "border": "#E1E9EF",
}

# ------------------------------------------------------------------
# Feature engineering assumption
# ------------------------------------------------------------------
# IMPORTANT: The uploaded imputer.pkl was fitted on 14 columns; the
# model expects 15 (it also uses "Smoking_Index"). That means
# Smoking_Index is engineered AFTER imputation, and the source
# notebook that defines its exact formula was not provided.
#
# ASSUMPTION (verify against your notebook and update if different):
#   Smoking_Index = Smoking_Years * Packs_Per_Year
#
# This mirrors the common "pack-year" style cumulative exposure index.
# Change the formula in utils/preprocessing.py -> compute_smoking_index()
# if your notebook defines it differently.
SMOKING_INDEX_ASSUMED = True

# ------------------------------------------------------------------
# Field definitions for the input form
# Each entry: (label, type, default, min, max, step, help, section)
# type is one of: "number", "int", "select"
# ------------------------------------------------------------------
FIELD_SECTIONS = {
    "Demographics": [
        {
            "key": "Age",
            "label": "Age (years)",
            "type": "int",
            "min": 10,
            "max": 100,
            "default": 30,
            "help": "Patient's current age in completed years.",
        },
        {
            "key": "Num_Pregnancies",
            "label": "Number of Pregnancies",
            "type": "int",
            "min": 0,
            "max": 20,
            "default": 1,
            "help": "Total number of pregnancies, including live births and losses.",
        },
    ],
    "Behavioral Risk Factors": [
        {
            "key": "Num_Sexual_Partners",
            "label": "Number of Sexual Partners",
            "type": "int",
            "min": 0,
            "max": 50,
            "default": 1,
            "help": "Total number of sexual partners over the patient's lifetime.",
        },
        {
            "key": "First_Sexual_Intercourse_Age",
            "label": "Age at First Sexual Intercourse",
            "type": "int",
            "min": 8,
            "max": 60,
            "default": 18,
            "help": "Age (years) at first sexual intercourse. Leave at default if not applicable.",
        },
        {
            "key": "Smoking_Years",
            "label": "Years Smoking",
            "type": "number",
            "min": 0.0,
            "max": 70.0,
            "default": 0.0,
            "step": 0.5,
            "help": "Total number of years the patient has smoked (0 if never smoked).",
        },
        {
            "key": "Packs_Per_Year",
            "label": "Packs per Year",
            "type": "number",
            "min": 0.0,
            "max": 50.0,
            "default": 0.0,
            "step": 0.5,
            "help": "Average number of cigarette packs smoked per year.",
        },
        {
            "key": "Hormonal_Contraceptive_Years",
            "label": "Hormonal Contraceptive Use (years)",
            "type": "number",
            "min": 0.0,
            "max": 40.0,
            "default": 0.0,
            "step": 0.5,
            "help": "Total years of hormonal contraceptive use (pills, injections, implants).",
        },
        {
            "key": "IUD_Years",
            "label": "IUD Use (years)",
            "type": "number",
            "min": 0.0,
            "max": 40.0,
            "default": 0.0,
            "step": 0.5,
            "help": "Total years of intrauterine device (IUD) use.",
        },
    ],
    "HPV / STD Information": [
        {
            "key": "Num_STDs",
            "label": "Number of STDs Diagnosed (lifetime)",
            "type": "int",
            "min": 0,
            "max": 20,
            "default": 0,
            "help": "Total number of distinct sexually transmitted diseases ever diagnosed.",
        },
        {
            "key": "STD_HPV",
            "label": "HPV Diagnosis",
            "type": "select",
            "options": ["No", "Yes"],
            "default": "No",
            "help": "Has the patient ever been diagnosed with HPV (Human Papillomavirus)?",
        },
        {
            "key": "STD_Num_Diagnoses",
            "label": "Number of STD Diagnoses (clinical visits)",
            "type": "int",
            "min": 0,
            "max": 20,
            "default": 0,
            "help": "Number of separate clinical STD diagnosis events on record.",
        },
    ],
    "Clinical Factors": [
        {
            "key": "Dx_Cancer",
            "label": "Prior Cancer Diagnosis",
            "type": "select",
            "options": ["No", "Yes"],
            "default": "No",
            "help": "Has the patient ever been diagnosed with any cancer?",
        },
        {
            "key": "Dx_CIN",
            "label": "Prior CIN Diagnosis",
            "type": "select",
            "options": ["No", "Yes"],
            "default": "No",
            "help": "Cervical Intraepithelial Neoplasia (CIN) - a precancerous condition.",
        },
        {
            "key": "Dx_HPV",
            "label": "Clinically Confirmed HPV Diagnosis",
            "type": "select",
            "options": ["No", "Yes"],
            "default": "No",
            "help": "Formal clinical diagnosis of HPV infection (distinct from self-reported STD_HPV).",
        },
    ],
}

# Flat lookup of all fields by key, useful for validation/preprocessing
ALL_FIELDS = {f["key"]: f for section in FIELD_SECTIONS.values() for f in section}

# Human-readable descriptions for SHAP / feature importance page
FEATURE_DESCRIPTIONS = {
    "Age": "Patient's age in years.",
    "Num_Sexual_Partners": "Lifetime number of sexual partners.",
    "First_Sexual_Intercourse_Age": "Age at first sexual intercourse.",
    "Num_Pregnancies": "Total number of pregnancies.",
    "Smoking_Years": "Number of years the patient has smoked.",
    "Packs_Per_Year": "Average cigarette packs smoked per year.",
    "Hormonal_Contraceptive_Years": "Years of hormonal contraceptive use.",
    "IUD_Years": "Years of IUD use.",
    "Num_STDs": "Number of STDs diagnosed in the patient's lifetime.",
    "STD_HPV": "Self-reported / historical HPV diagnosis (binary).",
    "STD_Num_Diagnoses": "Number of clinical STD diagnosis events.",
    "Dx_Cancer": "Prior cancer diagnosis (binary).",
    "Dx_CIN": "Prior CIN (precancerous lesion) diagnosis (binary).",
    "Dx_HPV": "Clinically confirmed HPV diagnosis (binary).",
    "Smoking_Index": "Engineered cumulative smoking exposure (Smoking_Years x Packs_Per_Year).",
}
