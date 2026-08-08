# Cervical Cancer Risk Predictor

A Streamlit web application for estimating cervical cancer risk from
demographic, behavioral, and clinical risk factors, built on top of a
trained XGBoost classifier.

⚠️ **Decision-support tool only. Not a diagnostic device.**

---

## 1. What's Included vs. What's Missing

This app was built from the model artifacts you uploaded:

| File | Status |
|---|---|
| `model.pkl` (XGBClassifier) | ✅ Included |
| `imputer.pkl` (IterativeImputer / MICE) | ✅ Included |
| `feature_names.pkl` | ✅ Included |
| `threshold.pkl` (0.20) | ✅ Included |
| `X_test.pkl` | ❌ Not provided |
| `y_test.pkl` | ❌ Not provided |
| `y_prob_test.pkl` | ❌ Not provided |
| `shap_explainer.pkl` | ❌ Not provided (app builds one at runtime instead) |
| Training notebook | ❌ Not provided |

**Impact:**
- **Prediction & SHAP explanation for a single patient work fully** —
  the app builds a `shap.TreeExplainer` from `model.pkl` directly, so
  `shap_explainer.pkl` isn't required.
- **Model Performance page** shows a clear "data not available"
  message instead of metrics, because it needs `X_test.pkl`,
  `y_test.pkl`, and `y_prob_test.pkl` to compute accuracy/precision/
  recall/F1/ROC AUC/confusion matrix honestly. **Drop those three
  files into `models/`** and the page will populate automatically —
  no code changes needed.
- **Global SHAP summary plot** (Feature Importance page, "Global"
  tab) needs a background dataset and uses `X_test.pkl` for that too.

### ⚠️ One assumption you must verify

Your `imputer.pkl` was fit on **14** columns, but `model.pkl` expects
**15** — the extra one is `Smoking_Index`, which must be engineered
*after* imputation. Since the training notebook wasn't provided, this
app assumes:

```
Smoking_Index = Smoking_Years * Packs_Per_Year
```

This lives in **one place**: `utils/preprocessing.py →
compute_smoking_index()`. If your notebook defines it differently,
edit that single function — nothing else needs to change. Until
verified, treat predictions as provisional.

---

## 2. Project Structure

```
cervical_cancer_app/
├── app.py                     # Home page (entrypoint)
├── pages/                     # Streamlit auto-discovers these for nav
│   ├── 2_ℹ️_About_the_Project.py
│   ├── 3_🩺_Predict_Risk.py
│   ├── 4_📊_Model_Performance.py
│   ├── 5_🔍_Feature_Importance.py
│   ├── 6_❓_FAQ.py
│   ├── 7_📚_References.py
│   └── 8_✉️_Contact.py
├── components/                 # Reusable UI building blocks
│   ├── cards.py                # Cards, metric rows, risk banner
│   ├── footer.py                # Shared footer + disclaimer
│   ├── sidebar.py                # Sidebar branding
│   └── styling.py                # CSS injection
├── utils/                      # Business logic (no UI code)
│   ├── config.py                # Paths, theme colors, form field defs
│   ├── model_loader.py           # Cached artifact loading + error handling
│   ├── preprocessing.py           # Impute -> engineer -> reorder pipeline
│   ├── prediction.py              # Threshold logic + risk interpretation
│   ├── validation.py               # Form validation rules
│   └── shap_utils.py                # SHAP explainer + explanation helpers
├── assets/
│   └── style.css                # Medical theme (dark blue/teal/green)
├── models/                      # Serialized artifacts (see table above)
├── .streamlit/config.toml        # Streamlit theme config
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

Business logic (`utils/`) is fully decoupled from UI code (`pages/`,
`components/`), so the preprocessing/prediction pipeline can be
unit-tested or reused (e.g. in a batch script) without Streamlit.

---

## 3. Running Locally

```bash
cd cervical_cancer_app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 4. Deploying to Streamlit Community Cloud

1. Push this folder to a **public or private GitHub repository**
   (root of the repo should be this folder, i.e. `app.py` at the top
   level of the repo, or set the app's entrypoint path accordingly).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click
   **New app**.
3. Select your repository, branch, and set **Main file path** to
   `app.py`.
4. Deploy. Streamlit Cloud will read `requirements.txt` and
   `runtime.txt` automatically.
5. **Model files:** `models/*.pkl` are committed to the repo by
   default (they're small). If you'd rather not commit binary model
   files to git, use [Git LFS](https://git-lfs.com/) or load them from
   a private storage bucket at startup instead — either approach only
   requires changing the paths in `utils/config.py`.

### Environment notes
- `xgboost` and `shap` both require a working C/C++ build toolchain
  on some platforms; Streamlit Cloud's default image supports both
  out of the box.
- If you see `InconsistentVersionWarning` from scikit-learn on
  startup, it means the artifacts were pickled with a different
  sklearn version than what's installed. It's a warning, not an
  error — but for reliability, pin `scikit-learn` in
  `requirements.txt` to the exact version used when training
  (`pip show scikit-learn` in your Colab notebook).

---

## 5. Security Notes

- The app performs **inference only** — no training data is loaded,
  displayed, or logged.
- Internal errors (stack traces, file paths) are never shown to the
  user; friendly messages are shown instead while full details go to
  the server-side log (`utils/model_loader.py`, try/except blocks in
  each page).
- Prediction inputs are session-scoped only — nothing is written to
  disk or an external database.
- The Contact page's form is a UI placeholder — wire it to a real
  email/CRM service before relying on it in production.

---

## 6. Testing Checklist

Before considering this deployment-ready, verify:

- [ ] `Smoking_Index` formula confirmed against the training notebook
      (see Section 1) and updated in `utils/preprocessing.py` if needed
- [ ] A known input/output pair from the Colab notebook reproduces the
      **same probability** in the Streamlit app (regression test)
- [ ] `X_test.pkl`, `y_test.pkl`, `y_prob_test.pkl` added and Model
      Performance page shows real metrics
- [ ] Form validation rejects out-of-range and logically inconsistent
      inputs (e.g. first-intercourse age > current age)
- [ ] SHAP waterfall/force plots render for at least 5 varied test
      cases (low, moderate, high risk)
- [ ] App behaves correctly with all default/zero values (a "healthy,
      never-smoked, no diagnoses" patient)
- [ ] Mobile/narrow-window layout checked (cards and columns collapse
      cleanly)
- [ ] Deployed app loads within Streamlit Cloud's free-tier resource
      limits (watch memory usage from SHAP + matplotlib)
- [ ] No tracebacks are ever shown to the end user (trigger an error
      deliberately, e.g. temporarily rename `model.pkl`, and confirm
      the friendly error message appears instead)

---

## 7. Optimization Suggestions

- **Cache SHAP background sampling**: the global summary plot samples
  up to 200 rows from `X_test`; increase/decrease
  `pages/5_🔍_Feature_Importance.py`'s `sample size` based on your
  Streamlit Cloud memory tier.
- **Precompute `shap_explainer.pkl`** in the notebook and include it —
  skips a small amount of runtime initialization, though for tree
  models this is already fast.
- **Batch prediction endpoint**: if you expect many patients at once,
  add a CSV-upload mode to the Predict page reusing
  `utils/preprocessing.py` and `utils/prediction.py` unchanged.
- **Model versioning**: store a version string alongside `model.pkl`
  (e.g. in `models/version.txt`) and surface it in the footer, so
  clinical users always know which model produced a given result.

---

## 8. Future Improvements

- Add authentication (e.g. `streamlit-authenticator`) before deploying
  for real patient use, given the sensitivity of the data involved.
- Add an audit log of predictions (with consent) for later model
  monitoring / drift detection.
- Localize the UI (multi-language support) for broader clinical reach.
- Add PDF export of a single patient's result + SHAP explanation for
  inclusion in a patient chart.
- Recalibrate/re-validate the model periodically as new labeled data
  becomes available (`sklearn.calibration.CalibratedClassifierCV`).

---

## Disclaimer

This application provides a statistical risk **estimate**, not a
medical diagnosis. All predictions must be reviewed by a qualified
healthcare professional alongside standard clinical screening
protocols before any decision is made.
