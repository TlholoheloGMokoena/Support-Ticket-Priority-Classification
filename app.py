"""Streamlit dashboard for the AI-Powered Support Ticket Classification System.   Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from support_ticket_classifier import DATA_PATH, MODEL_PATH
from train_model import run as train_run
from utils import analytics as an
from utils.prediction import load_model, predict_frame, predict_one
from utils.preprocessing import prepare_dataset, validate_columns

st.set_page_config(page_title="Ticket Triage", page_icon="🎫", layout="wide")
PRIORITY_COLORS = {"High": "#B42318", "Medium": "#B54708", "Low": "#027A48"}
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] {font-family: 'IBM Plex Sans', sans-serif;}
[data-testid="stMetric"] {background:#fff; border:1px solid #D9DEE7; border-left:4px solid #0F766E; border-radius:6px; padding:14px 16px;}
[data-testid="stMetric"] * {color:#16213A !important;}
[data-testid="stSidebar"] {background:#16213A;} [data-testid="stSidebar"] * {color:#E8ECF4 !important;}
.badge {display:inline-block; padding:6px 16px; border-radius:4px; color:#fff !important; font-weight:600; font-size:1.2rem;}
</style>""", unsafe_allow_html=True)


# ---------- cached loaders ----------
@st.cache_data
def load_dataset(path: str, mtime: float) -> pd.DataFrame:
    return prepare_dataset(pd.read_csv(path))


@st.cache_resource
def get_model(mtime: float):
    return load_model()


def get_dataset():
    if not DATA_PATH.exists():
        return None, f"Dataset not found at `data/Support_tickets.csv`. Add your CSV there or run `python generate_sample_data.py`."
    try:
        return load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime), ""
    except Exception as e:
        return None, f"Could not read the dataset: {e}"


def train_now():
    try:
        with st.spinner("Training model..."):
            train_run(DATA_PATH)
        st.cache_resource.clear()
        st.rerun()
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Training failed: {e}")


def require_model():
    bundle = get_model(MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0)
    if bundle is None:
        st.warning("No trained model found (or the model file could not be loaded).")
        if st.button("Train model now", type="primary"):
            train_now()
        st.stop()
    return bundle


def priority_badge(p):
    return f"<span class='badge' style='background:{PRIORITY_COLORS.get(p, '#475467')}'>{p}</span>"


# ---------- pages ----------
def page_overview():
    st.title("Dashboard overview")
    df, err = get_dataset()
    if df is None:
        st.error(err)
        return
    k = an.kpis(df)
    r1, r2 = st.columns(4), st.columns(4)
    r1[0].metric("Total tickets", f"{k['total']:,}")
    r1[1].metric("High priority", f"{k['High']:,}")
    r1[2].metric("Medium priority", f"{k['Medium']:,}")
    r1[3].metric("Low priority", f"{k['Low']:,}")
    r2[0].metric("Classified in history", f"{len(an.load_history()):,}")
    r2[1].metric("Most common category", k["top_category"])
    c1, c2 = st.columns(2)
    if "priority" in df:
        c1.subheader("Tickets by priority")
        c1.bar_chart(an.counts(df, "priority"))
    if "queue" in df:
        c2.subheader("Tickets by category")
        c2.bar_chart(an.counts(df, "queue"))
    c3, c4 = st.columns(2)
    if "type" in df:
        c3.subheader("Ticket types")
        c3.bar_chart(an.counts(df, "type"))
    daily = an.daily_volume(df)
    if daily is not None:
        c4.subheader("Ticket volume by day")
        c4.line_chart(daily)


def page_classify():
    st.title("Classify a new ticket")
    bundle = require_model()
    with st.form("single_ticket"):
        subject = st.text_input("Ticket subject")
        body = st.text_area("Ticket description", height=180)
        save = st.checkbox("Save to ticket history", value=True)
        go = st.form_submit_button("Classify ticket", type="primary")
    if not go:
        return
    try:
        res = predict_one(bundle, subject, body)
    except ValueError as e:
        st.warning(f"Please enter a subject or description. ({e})")
        return
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return
    a, b, c = st.columns(3)
    a.markdown("**Predicted priority**<br>" + priority_badge(res["predicted_priority"]), unsafe_allow_html=True)
    b.metric("Predicted category", res["predicted_category"] or "Not available")
    c.metric("Priority confidence", f"{res['priority_confidence']:.0%}")
    st.caption("Confidence is the model's probability estimate for its top class. It is not certainty and can be wrong.")
    st.info(res["explanation"])
    if save:
        an.save_tickets(pd.DataFrame([res]))
        st.success("Saved to ticket history.")


def page_bulk():
    st.title("Bulk ticket classification")
    bundle = require_model()
    st.caption("Upload a CSV with at least the columns `subject` and `body`.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            validate_columns(raw)
            if raw.empty:
                raise ValueError("The file contains no rows.")
            st.session_state["bulk"] = predict_frame(bundle, raw)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            st.error("This file could not be read as a CSV. Check the format and encoding (UTF-8).")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Classification failed: {e}")
    out = st.session_state.get("bulk")
    if out is None:
        return
    st.success(f"Classified {len(out):,} tickets.")
    st.dataframe(out)
    left, right = st.columns(2)
    left.download_button("Download classified CSV", out.to_csv(index=False).encode("utf-8"), "classified_tickets.csv", "text/csv")
    if right.button("Save all to ticket history"):
        st.success(f"Saved {an.save_tickets(out):,} tickets to history.")


def page_manage():
    st.title("Ticket management")
    hist = an.load_history()
    if hist.empty:
        st.info("No tickets saved yet. Classify a ticket and tick 'Save to ticket history'.")
        return
    f1, f2, f3 = st.columns([2, 1, 1])
    q = f1.text_input("Search subject or description")
    prio = f2.selectbox("Priority", ["All"] + an.PRIORITY_ORDER)
    cats = ["All"] + sorted(hist["predicted_category"].dropna().unique())
    cat = f3.selectbox("Category", cats)
    view = hist
    if q:
        view = view[view["subject"].str.contains(q, case=False, na=False) | view["body"].str.contains(q, case=False, na=False)]
    if prio != "All":
        view = view[view["predicted_priority"] == prio]
    if cat != "All":
        view = view[view["predicted_category"] == cat]
    st.caption(f"{len(view):,} of {len(hist):,} tickets")
    st.dataframe(view.drop(columns=["body"]), hide_index=True)
    if view.empty:
        return
    labels = dict(zip(view["id"], "#" + view["id"].astype(str) + " - " + view["subject"].astype(str)))
    t = view[view["id"] == st.selectbox("Open ticket", list(labels), format_func=labels.get)].iloc[0]
    st.subheader(t["subject"])
    st.write(t["body"])
    a, b, c = st.columns(3)
    a.markdown("**Priority**<br>" + priority_badge(t["predicted_priority"]), unsafe_allow_html=True)
    b.metric("Category", t["predicted_category"] or "n/a")
    c.metric("Classified at", t["classified_at"])


def show_model(res, title):
    m = res["metrics"]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision (weighted)", "precision"), ("Recall (weighted)", "recall"),
                                       ("F1 (weighted)", "f1"), ("F1 (macro)", "macro_f1")]):
        col.metric(label, f"{m[key]:.3f}")
    st.caption(f"Trained on {res['n_train']:,} tickets, tested on {res['n_test']:,} unseen tickets (80/20 split, fixed random state).")
    left, right = st.columns(2)
    left.pyplot(an.confusion_figure(res["cm"], res["labels"], f"{title}: confusion matrix"))
    rep = pd.DataFrame(res["report"]).T.round(3)
    right.dataframe(rep)
    per_class = rep.loc[res["labels"]]
    worst = per_class["recall"].idxmin()
    st.markdown(f"**Weakest class:** `{worst}` (recall {per_class.loc[worst, 'recall']:.2f}) - about "
                f"{1 - per_class.loc[worst, 'recall']:.0%} of real `{worst}` tickets were assigned another class. "
                "Read the confusion matrix rows as *actual* classes and columns as *predicted* classes.")


def page_performance():
    st.title("Model performance")
    bundle = require_model()
    if bundle.get("synthetic"):
        st.warning("This model was trained on the **synthetic sample dataset**. These numbers show the pipeline works, "
                   "not how it will perform on real tickets. Category accuracy of 1.0 is a symptom of simple synthetic text.")
    t1, t2 = st.tabs(["Priority model", "Category model"])
    with t1:
        show_model(bundle["priority"], "Priority")
        ev = bundle["priority"].get("rule_eval")
        if ev:
            st.subheader("Urgent-keyword rule: impact on High-priority detection")
            st.caption("The rule is separate from the model and never overrides it. This table shows what it would add if used.")
            st.dataframe(pd.DataFrame(ev).T.rename(index={"rule_only": "Rule only", "model_only": "Model only", "model_or_rule": "Model OR rule"}).round(3))
    with t2:
        if bundle["category"]:
            show_model(bundle["category"], "Category")
        else:
            st.info(bundle["category_note"])


# ---------- navigation ----------
PAGES = {"Dashboard": page_overview, "Classify a ticket": page_classify, "Bulk classification": page_bulk,
         "Ticket management": page_manage, "Model performance": page_performance}
st.sidebar.title("Ticket Triage")
choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
if st.sidebar.button("Retrain model"):
    train_now()
PAGES[choice]()
