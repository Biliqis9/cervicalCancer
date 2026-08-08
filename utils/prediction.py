"""
prediction.py
--------------
Wraps model.predict_proba() with the notebook's chosen decision
threshold (threshold.pkl) and produces a plain-language, color-coded
risk interpretation for the UI.
"""

from dataclasses import dataclass


@dataclass
class PredictionResult:
    probability: float
    predicted_class: int
    threshold: float
    risk_level: str  # "Low", "Moderate", "High"
    color: str  # "green", "yellow", "red"
    confidence: float
    explanation: str


def _risk_band(probability: float, threshold: float):
    """
    Three-band risk communication built around the model's own
    decision threshold, so the color coding always agrees with the
    predicted class:
      - Below threshold, closer to 0  -> Low (green)
      - Below threshold, but close to it -> Moderate (yellow)
      - At/above threshold -> High (red)
    The "close to it" buffer is 15% of the threshold's distance to 0/1,
    so the app never shows a green result the model actually flagged.
    """
    if probability >= threshold:
        return "High", "red"

    buffer = max(threshold * 0.5, 0.05)
    if probability >= threshold - buffer:
        return "Moderate", "yellow"

    return "Low", "green"


def interpret_prediction(probability: float, threshold: float) -> PredictionResult:
    predicted_class = int(probability >= threshold)
    risk_level, color = _risk_band(probability, threshold)

    # Confidence: distance from the threshold, scaled 0-100%, capturing
    # how decisively the model landed on its side of the cutoff.
    if predicted_class == 1:
        span = max(1.0 - threshold, 1e-6)
        confidence = min((probability - threshold) / span, 1.0)
    else:
        span = max(threshold, 1e-6)
        confidence = min((threshold - probability) / span, 1.0)
    confidence = max(confidence, 0.0) * 100

    explanations = {
        "Low": (
            "The model estimates a low likelihood of elevated cervical cancer risk "
            "based on the factors provided. This is not a diagnosis -- routine "
            "screening (e.g. Pap smear / HPV test) is still recommended per "
            "standard clinical guidelines."
        ),
        "Moderate": (
            "The model estimates a moderate likelihood of elevated risk. The "
            "result is close to the decision threshold, meaning small changes "
            "in the inputs could shift the outcome. Clinical follow-up and "
            "screening are recommended to confirm."
        ),
        "High": (
            "The model estimates an elevated likelihood of cervical cancer risk "
            "based on the factors provided. This indicates the case profile is "
            "similar to higher-risk patterns in the training data and warrants "
            "prompt clinical evaluation. This is a decision-support estimate, "
            "not a diagnosis."
        ),
    }

    return PredictionResult(
        probability=float(probability),
        predicted_class=predicted_class,
        threshold=float(threshold),
        risk_level=risk_level,
        color=color,
        confidence=float(confidence),
        explanation=explanations[risk_level],
    )


def predict(model, processed_df, threshold: float) -> PredictionResult:
    """
    Run inference on a single preprocessed row and return an
    interpreted PredictionResult. Errors are left to bubble up to the
    caller, which should catch them and show a friendly message
    (see pages/3_Predict_Risk.py).
    """
    proba = model.predict_proba(processed_df)[0, 1]
    return interpret_prediction(proba, threshold)
