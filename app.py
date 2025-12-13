import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------
# Opcionales (según entorno)
# -------------------------
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_OK = True
except Exception:
    NLTK_OK = False

try:
    from transformers import pipeline
    HF_OK = True
except Exception:
    HF_OK = False

try:
    from langdetect import detect
    LANGDETECT_OK = True
except Exception:
    LANGDETECT_OK = False


# =========================================================
# UI CONFIG + TERMINAL STYLE
# =========================================================
st.set_page_config(page_title="Survey & Bot Insights (TF-IDF + KMeans + Intents + Sentiment)", layout="wide")

terminal_style = """
<style>
.stApp { background-color: #000000; }
html, body, [class*="st-"], p, label, span, h1, h2, h3, h4, li {
    color: #00FF00 !important;
    font-family: "Courier New", monospace;
}
input, textarea, .stTextInput, .stFileUploader, .stSelectbox, .stMultiselect {
    background-color: #111 !important;
    color: #00FF00 !important;
    border: 1px solid #00FF00 !important;
}
div.stButton > button, .stDownloadButton button {
    background-color: #003300 !important;
    color: #00FF00 !important;
    border: 1px solid #00FF00;
    font-weight: bold;
}
.stProgress > div > div { background-color: #00FF00 !important; }
table, th, td {
    color: #00FF00 !important;
    background-color: #000000 !important;
    border: 1px solid #00FF00 !important;
}
svg { background-color: #000000 !important; }
</style>
"""
st.markdown(terminal_style, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def basic_clean(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_for_vectorizer(text: str) -> str:
    text = basic_clean(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_response_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if str(c).lower().startswith("respuesta")]
    # Orden “humano”: respuesta1, respuesta2, ... (si aplica)
    def keyfn(c):
        m = re.search(r"(\d+)", str(c))
        return int(m.group(1)) if m else 10**9
    cols.sort(key=keyfn)
    return cols

def build_text_clean(df: pd.DataFrame, resp_cols):
    # Concatena respuestas por fila
    texts = []
    for _, row in df[resp_cols].iterrows():
        parts = []
        for c in resp_cols:
            v = row.get(c, "")
            v = basic_clean(v)
            if v:
                parts.append(v)
        texts.append(" ".join(parts))
    return texts

def safe_sample(df: pd.DataFrame, n=15000):
    if len(df) <= n:
        return df
    return df.sample(n, random_state=42).reset_index(drop=True)

def fig_barh(labels, values, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

def top_terms_per_cluster(X, feature_names, labels, top_n=12):
    res = {}
    clusters = np.unique(labels)
    for k in clusters:
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        mean_tfidf = X[idx].mean(axis=0)
        arr = np.asarray(mean_tfidf).ravel()
        top_idx = arr.argsort()[::-1][:top_n]
        res[int(k)] = [(feature_names[i], float(arr[i])) for i in top_idx]
    return res

def deterministic_exec_summary(df_work: pd.DataFrame, k: int):
    lines = []
    lines.append(f"Total comentarios analizados: {len(df_work):,}")
    lines.append(f"Clusters (KMeans): {k}")
    if "sent_label" in df_work.columns:
        lines.append(f"Distribución sentimiento: {df_work['sent_label'].value_counts().to_dict()}")
    if "cluster" in df_work.columns:
        counts = df_work["cluster"].value_counts().sort_index().to_dict()
        lines.append(f"Distribución por cluster: {counts}")
    if "intent" in df_work.columns and df_work["intent"].notna().any():
        lines.append(f"Top intents (zero-shot): {df_work['intent'].value_counts().head(8).to_dict()}")
    return "\n".join(lines)

def build_llm_prompt_text(df_work: pd.DataFrame, top_terms: dict, top_n_clusters=6):
    parts = []
    parts.append(f"Dataset with {len(df_work)} customer survey comments.")
    if "sent_label" in df_work.columns:
        parts.append(f"Sentiment distribution: {df_work['sent_label'].value_counts().to_dict()}.")
    if "intent" in df_work.columns and df_work["intent"].notna().any():
        parts.append(f"Most common intents: {df_work['intent'].value_counts().head(10).to_dict()}.")

    # Clusters y ejemplos
    clusters = sorted(df_work["cluster"].unique().tolist())[:top_n_clusters]
    for cl in clusters:
        terms = ", ".join([t for t, _ in top_terms.get(int(cl), [])[:8]])
        ex = df_work[df_work["cluster"] == cl]["text_clean"].head(2).tolist()
        parts.append(f"Cluster {cl} keywords: {terms}. Examples: {' | '.join(ex)}")
        # Sentimiento por cluster
        if "sent_label" in df_work.columns:
            vc = df_work[df_work["cluster"] == cl]["sent_label"].value_counts().to_dict()
            parts.append(f"Cluster {cl} sentiment: {vc}.")

    return " ".join(parts)[:3200]

def detect_lang_fast(texts):
    if not LANGDETECT_OK:
        return "unknown"
    sample = " ".join([t for t in texts[:8] if isinstance(t, str)])[:5000]
    if not sample.strip():
        return "unknown"
    try:
        return detect(sample)
    except Exception:
        return "unknown"

# -------------------------
# Sentiment (robusto)
# -------------------------
@st.cache_resource(show_spinner=False)
def ensure_vader():
    if not NLTK_OK:
        return None
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

def sentiment_es_simple(text: str):
    # Heurística mínima (fallback)
    t = basic_clean(text).lower()
    pos = ["excelente", "bueno", "genial", "rápido", "recomiendo", "perfecto", "amable", "gracias", "sirvió"]
    neg = ["malo", "pésimo", "lento", "tarde", "horrible", "no sirve", "falló", "terrible", "caro", "no entiende"]
    p = sum(w in t for w in pos)
    n = sum(w in t for w in neg)
    score = 0.0 if (p + n) == 0 else (p - n) / (p + n)
    label = "pos" if score > 0.1 else ("neg" if score < -0.1 else "neu")
    return score, label

@st.cache_resource(show_spinner=False)
def load_hf_sentiment_safe():
    if not HF_OK:
        return None
    try:
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=-1)
    except Exception:
        return None

def hf_sentiment(pipe, text: str):
    out = pipe(basic_clean(text)[:512])[0]
    lab = out["label"].lower()
    conf = float(out["score"])
    if "neg" in lab:
        return -conf, "neg"
    if "pos" in lab:
        return conf, "pos"
    return 0.0, "neu"


# -------------------------
# Zero-shot (intents) seguro
# -------------------------
@st.cache_resource(show_spinner=False)
def load_zero_shot_safe(model_name: str):
    if not HF_OK:
        return None
    try:
        return pipeline("zero-shot-classification", model=model_name, device=-1)
    except Exception:
        return None


# -------------------------
# BART resumen seguro
# -------------------------
@st.cache_resource(show_spinner=False)
def load_bart_summarizer_safe():
    if not HF_OK:
        return None
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    except Exception:
        return None


# =========================================================
# APP HEADER
# =========================================================
st.title("📊 Survey & Bot Insights — Temas, Sentimiento, Intents y Recomendaciones")

st.markdown("""
Esta app analiza **respuestas abiertas** (encuestas / comentarios de bot) para:
- detectar **temas más hablados** (TF-IDF + KMeans),
- mapear a **intents/servicios** (Zero-shot opcional),
- medir **sentimiento**,
- y generar un **resumen ejecutivo** (determinístico + BART opcional).

✅ Diseñada para **no romperse** en Streamlit: si un modelo no carga, usa fallback.
""")

st.divider()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    uploaded = st.file_uploader("Sube tu CSV", type=["csv"])
    sep = st.selectbox("Separador", [",", ";", "\t", "|"], index=0)
    encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "latin-1"], index=1)

    st.divider()
    st.subheader("Muestreo / rendimiento")
    max_rows = st.slider("Máx filas a procesar", 500, 50000, 12000, step=500)

    st.divider()
    st.subheader("Temas emergentes (TF-IDF + KMeans)")
    k = st.slider("k (clusters)", 2, 20, 6, 1)
    ngrams = st.selectbox("N-gramas", ["(1,1)", "(1,2)", "(1,3)"], index=1)
    max_features = st.slider("Máx features TF-IDF", 2000, 50000, 20000, 1000)

    st.divider()
    st.subheader("Sentimiento")
    auto_lang = st.checkbox("Detectar idioma automáticamente", value=True)
    sentiment_mode = st.selectbox(
        "Método",
        ["auto", "HF multi (mejor, pesado)", "VADER (inglés rápido)", "ES simple (fallback)"],
        index=0
    )

    st.divider()
    st.subheader("Intents/Servicios (Zero-shot opcional)")
    use_zero_shot = st.checkbox("Activar zero-shot", value=True)
    zs_model = st.selectbox(
        "Modelo zero-shot",
        [
            "joeddav/xlm-roberta-base-xnli",  # multilingüe, suele ser más estable
            "facebook/bart-large-mnli",       # pesado, puede fallar
        ],
        index=0
    )
    default_intents = [
        "problemas de pago / facturación",
        "tiempo de respuesta / lentitud",
        "no entiende / mala comprensión del bot",
        "necesito agente humano / escalamiento",
        "problema técnico / falla / error",
        "seguimiento / estatus / tracking",
        "información incompleta / confusa",
        "queja general / mala experiencia",
        "felicitación / buena experiencia",
        "precio / costo",
    ]
    intents_text = st.text_area("Intents (uno por línea)", value="\n".join(default_intents), height=220)
    intents = [x.strip() for x in intents_text.splitlines() if x.strip()]
    hyp_template = st.text_input("Hypothesis template", value="This text is about {}.")

    st.divider()
    st.subheader("Resumen ejecutivo (LLM opcional)")
    use_bart = st.checkbox("Generar resumen con BART (si está disponible)", value=True)


# =========================================================
# LOAD DATA
# =========================================================
if not uploaded:
    st.info("👆 Sube un CSV para comenzar. Debe tener columnas tipo: respuesta1, respuesta2, ...")
    st.stop()

try:
    df0 = pd.read_csv(uploaded, sep=sep, encoding=encoding)
except Exception as e:
    st.error(f"No pude leer el CSV: {e}")
    st.stop()

if df0.empty:
    st.warning("El CSV está vacío.")
    st.stop()

df0 = safe_sample(df0, n=max_rows)

resp_cols = detect_response_columns(df0)
if not resp_cols:
    st.error("No encontré columnas que empiecen con 'respuesta'. Ejemplo esperado: respuesta1, respuesta2, ...")
    st.stop()

texts_clean = build_text_clean(df0, resp_cols)
df = df0.copy()
df["text_clean"] = texts_clean
df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

st.success(f"Filas a analizar: **{len(df):,}** | Columnas detectadas: **{len(resp_cols)}** ({', '.join(resp_cols[:6])}{'...' if len(resp_cols)>6 else ''})")


# =========================================================
# LANGUAGE
# =========================================================
lang = "unknown"
if auto_lang:
    lang = detect_lang_fast(df["text_clean"].tolist())
st.write(f"Idioma detectado (aprox): **{lang}**")


# =========================================================
# SENTIMENT
# =========================================================
vader = ensure_vader() if NLTK_OK else None
hf_sent_pipe = load_hf_sentiment_safe() if (HF_OK and ("HF" in sentiment_mode or sentiment_mode == "auto")) else None

def pick_sent_method():
    if sentiment_mode == "VADER (inglés rápido)":
        return "vader"
    if sentiment_mode == "ES simple (fallback)":
        return "es_simple"
    if sentiment_mode == "HF multi (mejor, pesado)":
        return "hf_multi"
    # auto
    if lang == "en" and vader is not None:
        return "vader"
    if hf_sent_pipe is not None:
        return "hf_multi"
    if lang == "es":
        return "es_simple"
    return "es_simple"

sent_method = pick_sent_method()

with st.spinner("Calculando sentimiento..."):
    sent_scores = []
    sent_labels = []
    for t in df["text_clean"].tolist():
        if sent_method == "vader" and vader is not None:
            s = vader.polarity_scores(t)["compound"]
            lab = "pos" if s >= 0.05 else ("neg" if s <= -0.05 else "neu")
            sent_scores.append(float(s))
            sent_labels.append(lab)
        elif sent_method == "hf_multi" and hf_sent_pipe is not None:
            s, lab = hf_sentiment(hf_sent_pipe, t)
            sent_scores.append(float(s))
            sent_labels.append(lab)
        else:
            s, lab = sentiment_es_simple(t)
            sent_scores.append(float(s))
            sent_labels.append(lab)

df["sent_score"] = sent_scores
df["sent_label"] = sent_labels


# =========================================================
# TF-IDF + KMeans
# =========================================================
norm_texts = [normalize_for_vectorizer(t) for t in df["text_clean"].tolist()]
ng = eval(ngrams)

with st.spinner("Calculando TF-IDF..."):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ng)
    X = vectorizer.fit_transform(norm_texts)
    feats = np.array(vectorizer.get_feature_names_out())

with st.spinner("Clustering con KMeans..."):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = km.fit_predict(X)
df["cluster"] = clusters

sil = None
try:
    if len(df) > k and k > 1:
        sil = silhouette_score(X, clusters)
except Exception:
    sil = None

top_terms = top_terms_per_cluster(X, feats, clusters, top_n=12)


# =========================================================
# ZERO-SHOT INTENTS (optional)
# =========================================================
df["intent"] = None
df["intent_score"] = None

zs = None
if use_zero_shot:
    with st.spinner("Cargando modelo zero-shot (si falla, se desactiva sin romper)..."):
        zs = load_zero_shot_safe(zs_model)

    if zs is None:
        st.warning("Zero-shot no disponible (modelo/descarga/memoria). Continuo sin intents.")
        use_zero_shot = False

if use_zero_shot and zs is not None and intents:
    batch = 8
    texts = df["text_clean"].tolist()
    intents_out = []
    scores_out = []
    with st.spinner("Clasificando intents con zero-shot..."):
        for i in range(0, len(texts), batch):
            bt = texts[i:i+batch]
            try:
                res = zs(bt, intents, hypothesis_template=hyp_template)
            except Exception:
                st.warning("Zero-shot falló durante ejecución. Se desactiva y se conserva lo ya hecho.")
                break

            if isinstance(res, dict):
                res = [res]
            for r in res:
                intents_out.append(r["labels"][0])
                scores_out.append(float(r["scores"][0]))

    n_done = len(intents_out)
    if n_done > 0:
        df.loc[:n_done-1, "intent"] = intents_out
        df.loc[:n_done-1, "intent_score"] = scores_out


# =========================================================
# DASHBOARD
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Comentarios", f"{len(df):,}")
c2.metric("Clusters", f"{k}")
c3.metric("Features TF-IDF", f"{X.shape[1]:,}")
c4.metric("Silhouette", f"{sil:.3f}" if sil is not None else "N/A")

st.divider()

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("📌 Temas emergentes (clusters)")
    cl_counts = df["cluster"].value_counts().sort_index()
    st.pyplot(fig_barh(cl_counts.index.astype(str), cl_counts.values, "Distribución por cluster"))

    st.subheader("🙂 Sentimiento (global)")
    s_counts = df["sent_label"].value_counts()
    st.pyplot(fig_barh(s_counts.index.astype(str), s_counts.values, "Distribución de sentimiento"))

    st.subheader("😡 Dónde duele más (sentimiento por cluster)")
    grp = df.groupby("cluster")["sent_score"].mean().sort_values()
    st.pyplot(fig_barh(grp.index.astype(str), grp.values, "Sentimiento promedio por cluster (más negativo abajo)"))

    if use_zero_shot and df["intent"].notna().any():
        st.subheader("🧭 Intents/servicios (zero-shot)")
        ic = df["intent"].value_counts().head(12)
        st.pyplot(fig_barh(ic.index.astype(str), ic.values, "Top intents (zero-shot)"))

        st.subheader("🔥 Intents con peor sentimiento (promedio)")
        ig = df.groupby("intent")["sent_score"].mean().sort_values().head(10)
        st.pyplot(fig_barh(ig.index.astype(str), ig.values, "Peor sentimiento por intent"))

with right:
    st.subheader("🔑 Keywords por cluster")
    for cl in sorted(top_terms.keys()):
        terms = ", ".join([t for t, _ in top_terms[cl][:10]])
        st.markdown(f"**Cluster {cl}**: {terms}")

    st.subheader("🧪 Ejemplos")
    chosen_cluster = st.selectbox("Cluster", sorted(df["cluster"].unique().tolist()))
    n_ex = st.slider("Ejemplos a mostrar", 3, 30, 10)
    show_cols = ["text_clean", "sent_label", "sent_score"]
    if use_zero_shot:
        show_cols += ["intent", "intent_score"]
    st.dataframe(df[df["cluster"] == chosen_cluster][show_cols].head(n_ex), use_container_width=True)

st.divider()


# =========================================================
# BOT RECOMMENDATIONS (heurísticas útiles)
# =========================================================
st.subheader("🛠️ Recomendaciones para mejorar el bot (heurísticas)")

# 1) clusters más negativos
neg_clusters = df.groupby("cluster")["sent_score"].mean().sort_values().head(3)
st.markdown("**Clusters con peor sentimiento promedio (prioridad alta):**")
for cl, sc in neg_clusters.items():
    st.markdown(f"- Cluster **{cl}** → sent_score_prom: **{sc:.2f}**")

# 2) intents más negativos y frecuentes
if use_zero_shot and df["intent"].notna().any():
    tmp = df.dropna(subset=["intent"]).copy()
    freq = tmp["intent"].value_counts()
    neg = tmp.groupby("intent")["sent_score"].mean()
    score = (freq.reindex(neg.index).fillna(0) * 0.6) + ((-neg).clip(lower=0) * 100 * 0.4)
    top_actions = score.sort_values(ascending=False).head(8)

    st.markdown("**Intents/servicios a revisar (frecuencia + negatividad):**")
    for intent_name in top_actions.index.tolist():
        st.markdown(f"- **{intent_name}** | freq={int(freq.get(intent_name, 0))} | sent_avg={float(neg.get(intent_name, 0)):.2f}")
else:
    st.info("Activa zero-shot si quieres recomendaciones por intent/servicio (opcional).")

st.divider()


# =========================================================
# EXPORT
# =========================================================
st.subheader("⬇️ Exportar resultados")

export_cols = ["text_clean", "cluster", "sent_label", "sent_score"]
if use_zero_shot:
    export_cols += ["intent", "intent_score"]

out_df = df[export_cols].copy()
csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig", errors="ignore")

st.download_button(
    "Descargar CSV con resultados",
    data=csv_bytes,
    file_name="survey_bot_insights.csv",
    mime="text/csv"
)

st.caption("Muestra (primeras 30 filas):")
st.dataframe(out_df.head(30), use_container_width=True)

st.divider()


# =========================================================
# EXEC SUMMARY (Deterministic + BART optional)
# =========================================================
st.subheader("🧠 Resumen ejecutivo")

st.markdown("**Resumen determinístico (siempre disponible):**")
st.code(deterministic_exec_summary(df, k))

if use_bart:
    bart = load_bart_summarizer_safe()
    if bart is None:
        st.info("ℹ️ BART no está disponible (descarga/memoria). Se mantiene el resumen determinístico.")
    else:
        try:
            llm_text = build_llm_prompt_text(df, top_terms)
            with st.spinner("Generando resumen con BART..."):
                summ = bart(llm_text, max_length=190, min_length=90, do_sample=False)[0]["summary_text"]
            st.markdown("**Resumen con BART (opcional):**")
            st.write(summ)
        except Exception:
            st.warning("⚠️ BART falló en ejecución. Se mantiene el resumen determinístico.")
