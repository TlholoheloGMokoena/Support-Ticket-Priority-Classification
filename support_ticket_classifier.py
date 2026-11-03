"""Core ML module: pipeline definition, training, evaluation and the (separate) rule-based urgency flag."""
import re
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent          # relative to the project, so it works when deployed
DATA_PATH = BASE_DIR / "data" / "Support_tickets.csv"
MODEL_PATH = BASE_DIR / "models" / "support_ticket_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42
MIN_PER_CLASS = 20  # classes with fewer examples are too thin to learn or evaluate reliably

# Rule-based urgency flag. It is informational only: it NEVER changes the model's prediction.
URGENT_PATTERN = re.compile(r"\b(outage|down|hacked|breach|compromised|data loss|lost data|crash(?:es|ed)?|urgent|asap|unauthori[sz]ed|security incident)\b")


def rule_flag(text: str) -> bool:
    return bool(URGENT_PATTERN.search(text))


def build_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression. class_weight='balanced' compensates for imbalanced classes."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def train_target(df: pd.DataFrame, target: str):
    """Train and evaluate one model. Returns (result_dict, note); result_dict is None if the labels are unusable."""
    if target not in df.columns:
        return None, f"Column '{target}' not found in the dataset."
    d = df.dropna(subset=[target])
    counts = d[target].value_counts()
    d = d[d[target].isin(counts[counts >= MIN_PER_CLASS].index)]
    if d[target].nunique() < 2:
        return None, f"'{target}' has fewer than 2 classes with at least {MIN_PER_CLASS} tickets, so no reliable model was trained."
    X_tr, X_te, y_tr, y_te = train_test_split(d["text"], d[target].astype(str), test_size=0.2, random_state=RANDOM_STATE, stratify=d[target])
    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    labels = list(pipe.classes_)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    result = {
        "pipeline": pipe, "labels": labels, "n_train": len(X_tr), "n_test": len(X_te),
        "class_counts": d[target].value_counts().to_dict(),
        "metrics": {"accuracy": accuracy_score(y_te, pred), "precision": p, "recall": r, "f1": f,
                    "macro_f1": precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)[2]},
        "report": classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0),
        "cm": confusion_matrix(y_te, pred, labels=labels).tolist(),
    }
    if target == "priority" and "High" in labels:  # measure what the rule flag adds, on the same test set
        flag, is_high, model_high = X_te.map(rule_flag), (y_te == "High"), pd.Series(pred == "High", index=y_te.index)
        def pr(pos):
            tp = (pos & is_high).sum()
            return {"precision": float(tp / pos.sum()) if pos.sum() else 0.0, "recall": float(tp / is_high.sum())}
        result["rule_eval"] = {"rule_only": pr(flag), "model_only": pr(model_high), "model_or_rule": pr(flag | model_high)}
    return result, ""


def train_all(df: pd.DataFrame) -> dict:
    priority, p_note = train_target(df, "priority")
    if priority is None:
        raise ValueError(f"Priority model could not be trained: {p_note}")
    category, c_note = train_target(df, "queue")
    return {"priority": priority, "category": category, "category_note": c_note or "Category = `queue` column.",
            "n_tickets": len(df), "synthetic": "data_source" in df.columns and (df["data_source"] == "synthetic-sample").all()}


def save_model(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
