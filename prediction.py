"""Load the saved model and classify one ticket or a whole DataFrame."""
import joblib
import numpy as np
import pandas as pd
from support_ticket_classifier import MODEL_PATH, rule_flag
from utils.preprocessing import combine_text, validate_columns


def load_model(path=MODEL_PATH):
    """Return the saved model bundle, or None if it is missing or unreadable."""
    try:
        return joblib.load(path) if path.exists() else None
    except Exception:
        return None


def _predict(res, texts):
    proba = res["pipeline"].predict_proba(texts)
    return res["pipeline"].classes_[proba.argmax(axis=1)], proba.max(axis=1)


def predict_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add predicted_priority/category, confidences and the informational urgent-keyword flag."""
    validate_columns(df)
    out = df.copy()
    text = combine_text(out)
    if (text == "").any():
        raise ValueError(f"{int((text == '').sum())} ticket(s) have an empty subject and description.")
    out["predicted_priority"], out["priority_confidence"] = _predict(bundle["priority"], text)
    if bundle.get("category"):
        out["predicted_category"], out["category_confidence"] = _predict(bundle["category"], text)
    else:
        out["predicted_category"], out["category_confidence"] = None, np.nan
    out["urgent_keyword_flag"] = text.map(rule_flag)
    return out


def top_terms(res, text: str, label: str, n: int = 5) -> list:
    """Words that pushed the model toward `label` (TF-IDF weight x coefficient)."""
    vec, clf = res["pipeline"].named_steps["tfidf"], res["pipeline"].named_steps["clf"]
    x = vec.transform([text]).toarray()[0]
    idx = list(clf.classes_).index(label)
    coef = clf.coef_[idx] if clf.coef_.shape[0] > 1 else (clf.coef_[0] if idx == 1 else -clf.coef_[0])
    names = vec.get_feature_names_out()
    order = np.argsort(x * coef)[::-1][:n]
    return [names[i] for i in order if x[i] * coef[i] > 0]


def predict_one(bundle: dict, subject: str, body: str) -> dict:
    """Classify a single ticket and return the prediction plus a short plain-language explanation."""
    one = pd.DataFrame([{"subject": subject, "body": body}])
    row = predict_frame(bundle, one).iloc[0].to_dict()
    terms = top_terms(bundle["priority"], combine_text(one).iloc[0], row["predicted_priority"])
    expl = f"The model predicts **{row['predicted_priority']}** priority"
    expl += f" mainly because of the terms: {', '.join(terms)}." if terms else ", but no single term stood out."
    if row["urgent_keyword_flag"] and row["predicted_priority"] != "High":
        expl += " Urgent keywords were also detected (rule-based, informational only); the model's prediction was not changed."
    row["explanation"] = expl
    return row
