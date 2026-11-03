"""Text cleaning and dataset preparation (shared by training and prediction so both see identical text)."""
import re
import pandas as pd

REQUIRED_COLUMNS = ["subject", "body"]


def clean_text(text) -> str:
    """Lowercase, strip punctuation/special characters and collapse whitespace."""
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_columns(df: pd.DataFrame, required=REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Found: {', '.join(df.columns)}")


def combine_text(df: pd.DataFrame) -> pd.Series:
    """Merge subject + body into one cleaned text field. Answer/response columns are deliberately NOT used (data leakage)."""
    validate_columns(df)
    return (df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)).map(clean_text)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicates, normalise labels and add the `text` column."""
    validate_columns(df)
    df = df.drop_duplicates(subset=["subject", "body"]).copy()
    df["text"] = combine_text(df)
    df = df[df["text"] != ""]
    for col in ["priority", "queue", "type"]:  # normalise label spelling: " high " -> "High"
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.title()
    return df.reset_index(drop=True)
