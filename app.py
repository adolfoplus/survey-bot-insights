# =========================================================
# STREAMLIT SAFE BOOT
# =========================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # evita CUDA init
HF_OK = False  # transformers DESACTIVADO al inicio (CRÍTICO)

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Sentiment básico (siempre disponible)
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon")
    vader = SentimentIntensityAnalyzer()
    NLTK_OK = True
except Exception:
    NLTK_OK = False
    vader = None

# =========================================================
# UI CONFIG
# =========================================================
st.set_page_config(
    page_title="Survey Bot Insights",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #000000; }
html, body, [class*="st-"], p, label, span, h1, h2, h3 {
    color: #00FF00 !important;
    font-family: "Courier New", monospace;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Survey Bot Insights")
st.caption("Análisis de comentarios abiertos para mejorar bots e intents")

# =========================================================
# HELPERS
# =========================================================
def detect_response_columns(df):
    cols = [c for c in df.columns if str(c).lower().startswith("respuesta")]
    cols.sort()
    return cols

def clean_text(row, cols):
    parts = []
    for c in cols:
        v = str(row.get(c, "")).strip()
        if v:
            parts.append(v)
    return " ".join(parts)

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentiment(text):
    if vader:
        s = vader.polarity_scores(text)["compound"]
        lab = "positivo" if s >= 0.05 else ("negativo" if s <= -0.05 else "neutral")
        return s, lab
    return 0.0, "neutral"

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    uploaded = st.file_uploader("Sube tu CSV", type=["csv"])
    sep = st.selectbox("Separador", [",", ";", "|", "\t"], index=0)

    k = st.slider("Número de clusters (temas)", 2, 12, 5)
    max_features = st.slider("TF-IDF features", 2000, 20000, 8000)

    use_llm = st.checkbox(
        "Activar análisis avanzado (Zero-shot / LLM)",
        value=False,
        help="Se carga SOLO después de subir CSV"
    )

# =========================================================
# MAIN FLOW
# =========================================================
if not uploaded:
    st.info("⬅️ Sube un CSV con columnas tipo: respuesta1, respuesta2, ...")
    st.stop()

# 🔓 AHORA sí permitimos transformers si el usuario lo pide
if use_llm:
    HF_OK = True

# Cargar CSV
df_raw = pd.read_csv(uploaded, sep=sep)
resp_cols = detect_response_columns(df_raw)

if not resp_cols:
    st.error("No se encontraron columnas 'respuestaX'")
    st.stop()

df = df_raw.copy()
df["text"] = df.apply(lambda r: clean_text(r, resp_cols), axis=1)
df = df[df["text"].str.len() > 0].reset_index(drop=True)

st.success(f"Filas analizadas: {len(df)}")

# =========================================================
# SENTIMENT
# =========================================================
with st.spinner("Analizando sentimiento..."):
    df["sent_score"], df["sent_label"] = zip(*df["text"].map(sentiment))

# =========================================================
# TF-IDF + KMEANS
# =========================================================
with st.spinner("Detectando temas (TF-IDF + KMeans)..."):
    texts_norm = df["text"].map(normalize).tolist()
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(texts_norm)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(X)

# =========================================================
# DASHBOARD
# =========================================================
c1, c2, c3 = st.columns(3)
c1.metric("Comentarios", len(df))
c2.metric("Clusters", k)
c3.metric("Sentimiento negativo (%)", f"{(df['sent_label']=='negativo').mean()*100:.1f}%")

st.divider()

# Clusters
st.subheader("📌 Temas detectados")
cluster_counts = df["cluster"].value_counts().sort_index()
fig, ax = plt.subplots()
ax.barh(cluster_counts.index.astype(str), cluster_counts.values)
ax.invert_yaxis()
st.pyplot(fig)

# Sentimiento por cluster
st.subheader("😡 Sentimiento promedio por tema")
grp = df.groupby("cluster")["sent_score"].mean().sort_values()
fig2, ax2 = plt.subplots()
ax2.barh(grp.index.astype(str), grp.values)
ax2.invert_yaxis()
st.pyplot(fig2)

# Ejemplos
st.subheader("🧪 Ejemplos por tema")
chosen = st.selectbox("Cluster", sorted(df["cluster"].unique()))
st.dataframe(
    df[df["cluster"] == chosen][["text", "sent_label"]].head(15),
    use_container_width=True
)

# =========================================================
# EXPORT
# =========================================================
st.divider()
st.subheader("⬇️ Exportar resultados")

out = df[["text", "cluster", "sent_label", "sent_score"]]
csv = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

st.download_button(
    "Descargar CSV",
    data=csv,
    file_name="survey_bot_insights_results.csv",
    mime="text/csv"
)

# =========================================================
# OPTIONAL LLM SUMMARY (SAFE)
# =========================================================
if use_llm and HF_OK:
    try:
        from transformers import pipeline
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1
        )

        st.divider()
        st.subheader("🧠 Resumen ejecutivo (LLM)")

        summary_text = (
            f"Total comentarios: {len(df)}. "
            f"Clusters: {cluster_counts.to_dict()}. "
            f"Sentimiento: {df['sent_label'].value_counts().to_dict()}."
        )

        res = summarizer(summary_text, max_length=120, min_length=50, do_sample=False)
        st.write(res[0]["summary_text"])

    except Exception:
        st.warning("LLM no disponible en este entorno. Se usó análisis base.")
