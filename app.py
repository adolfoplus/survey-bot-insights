# =========================================================
# STREAMLIT SAFE BOOT
# =========================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HF_OK = False

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sentiment básico
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
# CLEAN GOOGLE-LIKE STYLE
# =========================================================
st.markdown("""
<style>
html, body, .stApp {
    background-color: #ffffff;
    color: #202124;
    font-family: "Roboto", "Arial", sans-serif;
}

h1 {
    font-size: 2.4rem;
    font-weight: 600;
}

h2 {
    font-size: 1.6rem;
    margin-top: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #e0e0e0;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 12px;
    background: #f1f3f4;
    text-align: center;
}

.stDataFrame {
    background: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("Survey Bot Insights")
st.caption("Analyze open-ended survey responses to improve bots, intents, and services")

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

    k = st.slider("Number of topics (clusters)", 2, 12, 5)
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

# =========================================================
# ANALYSIS
# =========================================================
df["sent_score"], df["sentiment"] = zip(*df["text"].map(sentiment))

texts_norm = df["text"].map(normalize)
tfidf = TfidfVectorizer(max_features=max_features)
X = tfidf.fit_transform(texts_norm)

km = KMeans(n_clusters=k, random_state=42, n_init="auto")
df["topic"] = km.fit_predict(X)

# =========================================================
# METRICS
# =========================================================
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Comments analyzed", len(df))

with m2:
    st.metric("Topics detected", k)

with m3:
    st.metric(
        "Negative sentiment",
        f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%"
    )

st.divider()

# =========================================================
# VISUALS
# =========================================================
st.subheader("Detected topics")

topic_counts = df["topic"].value_counts().sort_index()
fig, ax = plt.subplots()
ax.barh(topic_counts.index.astype(str), topic_counts.values)
ax.set_xlabel("Number of comments")
st.pyplot(fig)

st.subheader("Average sentiment by topic")

grp = df.groupby("topic")["sent_score"].mean().sort_values()
fig2, ax2 = plt.subplots()
ax2.barh(grp.index.astype(str), grp.values)
ax2.set_xlabel("Average sentiment score")
st.pyplot(fig2)

# =========================================================
# EXAMPLES
# =========================================================
st.subheader("Example comments")

chosen = st.selectbox("Select topic", sorted(df["topic"].unique()))
st.dataframe(
    df[df["topic"] == chosen][["text", "sentiment"]].head(15),
    use_container_width=True
)

# =========================================================
# EXPORT
# =========================================================
st.divider()
st.subheader("Export results")

out = df[["text", "topic", "sentiment", "sent_score"]]
csv = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

st.download_button(
    "Download CSV",
    data=csv,
    file_name="survey_bot_insights_results.csv",
    mime="text/csv"
)
