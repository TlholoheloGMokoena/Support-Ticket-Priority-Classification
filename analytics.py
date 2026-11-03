"""Dashboard statistics, confusion-matrix figure and SQLite ticket history."""
import sqlite3
from contextlib import closing
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from support_ticket_classifier import OUTPUT_DIR

DB_PATH = OUTPUT_DIR / "ticket_history.db"
PRIORITY_ORDER = ["High", "Medium", "Low"]
HIST_COLS = ["id", "classified_at", "subject", "body", "predicted_priority", "priority_confidence",
             "predicted_category", "category_confidence", "urgent_keyword_flag"]


def counts(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].value_counts()
    return s.reindex([p for p in PRIORITY_ORDER if p in s.index]) if col == "priority" else s


def kpis(df: pd.DataFrame) -> dict:
    pr = df["priority"].value_counts() if "priority" in df else pd.Series(dtype=int)
    top = df["queue"].value_counts().idxmax() if "queue" in df and len(df) else "n/a"
    return {"total": len(df), "High": int(pr.get("High", 0)), "Medium": int(pr.get("Medium", 0)), "Low": int(pr.get("Low", 0)), "top_category": top}


def daily_volume(df: pd.DataFrame, col: str = "created_at"):
    if col not in df:
        return None
    d = pd.to_datetime(df[col], errors="coerce").dropna()
    return d.dt.date.value_counts().sort_index() if len(d) else None


def confusion_figure(cm, labels, title="Confusion matrix (test set)"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


# ---- ticket history (SQLite, stored in outputs/) ----
def _connect():
    OUTPUT_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, classified_at TEXT, subject TEXT, body TEXT,
                   predicted_priority TEXT, priority_confidence REAL, predicted_category TEXT, category_confidence REAL, urgent_keyword_flag INTEGER)""")
    return con


def _py(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def save_tickets(df: pd.DataFrame) -> int:
    d = df.reindex(columns=HIST_COLS[2:]).astype(object)
    now = datetime.now().isoformat(timespec="seconds")
    recs = [(now, *[_py(v) for v in r]) for r in d.itertuples(index=False)]
    with closing(_connect()) as con, con:
        con.executemany(f"INSERT INTO tickets ({','.join(HIST_COLS[1:])}) VALUES ({','.join('?' * (len(HIST_COLS) - 1))})", recs)
    return len(recs)


def load_history() -> pd.DataFrame:
    with closing(_connect()) as con:
        h = pd.read_sql_query("SELECT * FROM tickets ORDER BY id DESC", con)
    h["urgent_keyword_flag"] = h["urgent_keyword_flag"].astype(bool)
    return h
