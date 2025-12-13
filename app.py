# =========================================================
# SAFE BOOT (Streamlit Cloud)
# =========================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sentiment (VADER – ligero y estable)
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon")
    vader = SentimentIntensityAnalyzer()
except Exception:
    vader = None

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Survey Bot Insights",
    layout="wide"
)

# =========================================================
# CLAUDE-STYLE UI
# =========================================================
st.markdown("""
<style>
html, body, .stApp {
    background-color: #FAFAFA;
    color: #1F1F1F;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1 { font-size: 2.3rem; font-weight: 600; }
h2 { font-size: 1.6rem; font-weight: 500; margin-top: 2rem; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

.stMetric {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1.2rem;
}

.stDataFrame {
    background: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
}

hr {
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("Survey Bot Insights")
st.caption("Understand what users are really saying and where your bot needs improvement")

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.subheader("Configuration")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    sep = st.selectbox("CSV separator", [",", ";", "|", "\t"], index=0)

    st.divider()
    st.subheader("Topic detection")

    k = st.slider("Number of topics", 2, 12, 5)
    max_features = st.slider("TF-IDF features", 2000, 20000, 8000)

# =========================================================
# HELPERS
# =========================================================
def detect_response_columns(df):
    return [c for c in df.columns if c.lower().startswith("respuesta")]

def clean_text(row, cols):
    return " ".join(str(row[c]) for c in cols if str(row[c]).strip())

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def sentiment(text):
    if vader:
        s = vader.polarity_scores(text)["compound"]
        lab = "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
        return s, lab
    return 0.0, "Neutral"

# =========================================================
# TOPIC LABELING
# =========================================================
def get_topic_keywords(X, labels, feature_names, top_n=5):
    out = {}
    for t in np.unique(labels):
        idx = np.where(labels == t)[0]
        mean = X[idx].mean(axis=0).A1
        top = mean.argsort()[::-1][:top_n]
        out[t] = ", ".join(feature_names[i] for i in top)
    return out

TOPIC_RULES = {
    "Payment & Billing": ["pago", "factura", "cobro", "precio", "tarjeta", "dinero"],
    "Response Time": ["lento", "tarda", "espera", "demora"],
    "Human Agent Request": ["humano", "agente", "asesor", "persona"],
    "Technical Errors": ["error", "falla", "bug", "problema", "no funciona"],
    "Account & Login": ["login", "cuenta", "acceso", "password"],
    "Delivery & Orders": ["envío", "entrega", "pedido"]
}

def human_topic_name(keywords):
    kws = keywords.lower()
    for name, terms in TOPIC_RULES.items():
        if any(t in kws for t in terms):
            return name
    return "Other Issues"

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================
def executive_summary(df):
    total = len(df)
    neg = (df["sentiment"] == "Negative").mean() * 100

    top = df["topic_name"].value_counts().head(3)
    worst = df.groupby("topic_name")["sent_score"].mean().sort_values().head(2)

    text = f"""
**Overview**

We analyzed **{total} user comments**.

**Most frequent topics**
"""
    for t, c in top.items():
        text += f"- **{t}** ({c} comments)\n"

    text += f"\n**Negative sentiment:** {neg:.1f}%\n\n"
    text += "**Topics generating most frustration**\n"
    for t, s in worst.items():
        text += f"- **{t}** (avg sentiment {s:.2f})\n"

    text += """
**Recommendation**
Prioritize these topics for bot improvements, new intents,
or escalation to human agents.
"""
    return text

# =========================================================
# NEW INTENT DETECTION
# =========================================================
def detect_new_intents(df, min_comments=3):
    g = df.groupby("topic_name").agg(
        comments=("text", "count"),
        avg_sentiment=("sent_score", "mean")
    ).reset_index()

    return g[
        (g.comments >= min_comments) &
        (g.avg_sentiment < -0.1)
    ].sort_values("avg_sentiment")

# =========================================================
# MAIN FLOW
# =========================================================
if not uploaded:
    st.info("Upload a CSV with columns like respuesta1, respuesta2, ...")
    st.stop()

df_raw = pd.read_csv(uploaded, sep=sep)
resp_cols = detect_response_columns(df_raw)

if not resp_cols:
    st.error("No columns starting with 'respuesta' were found.")
    st.stop()

df = df_raw.copy()
df["text"] = df.apply(lambda r: clean_text(r, resp_cols), axis=1)
df = df[df["text"].str.len() > 0].reset_index(drop=True)

df["sent_score"], df["sentiment"] = zip(*df["text"].map(sentiment))

texts_norm = df["text"].map(normalize)
tfidf = TfidfVectorizer(max_features=max_features)
X = tfidf.fit_transform(texts_norm)

km = KMeans(n_clusters=k, random_state=42, n_init="auto")
df["topic"] = km.fit_predict(X)

keywords = get_topic_keywords(X, df["topic"].values, tfidf.get_feature_names_out())
df["topic_name"] = df["topic"].map(lambda t: human_topic_name(keywords[t]))

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Insights", "Examples"])

# ---------------- Overview ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Comments analyzed", len(df))
    c2.metric("Topics detected", df["topic_name"].nunique())
    c3.metric("Negative sentiment", f"{(df['sentiment']=='Negative').mean()*100:.1f}%")

    st.subheader("Detected topics")
    counts = df["topic_name"].value_counts()
    fig, ax = plt.subplots()
    ax.barh(counts.index, counts.values)
    ax.invert_yaxis()
    st.pyplot(fig)

# ---------------- Insights ----------------
with tab2:
    st.subheader("Executive summary")
    st.markdown(executive_summary(df))

    st.subheader("Suggested new intents")
    intents = detect_new_intents(df)
    if intents.empty:
        st.success("No critical missing intents detected.")
    else:
        for _, r in intents.iterrows():
            st.markdown(
                f"- **{r['topic_name']}** → {r['comments']} comments, "
                f"avg sentiment {r['avg_sentiment']:.2f}"
            )

# ---------------- Examples ----------------
with tab3:
    chosen = st.selectbox("Select topic", sorted(df["topic_name"].unique()))
    st.dataframe(
        df[df["topic_name"] == chosen][["text", "sentiment"]],
        use_container_width=True
    )

# =========================================================
# EXPORT
# =========================================================
st.divider()
out = df[["text", "topic_name", "sentiment", "sent_score"]]
st.download_button(
    "Download results CSV",
    out.to_csv(index=False, encoding="utf-8-sig"),
    "survey_bot_insights_results.csv",
    "text/csv"
)
