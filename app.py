# =========================================================
# SAFE BOOT (Streamlit Cloud)
# =========================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import re
import importlib.util
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, silhouette_score)

HAS_PYSENTIMIENTO = importlib.util.find_spec("pysentimiento") is not None
LABELS = ["Positive", "Neutral", "Negative"]

# =========================================================
# TEXT UTILITIES
# =========================================================
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def is_empty(v) -> bool:
    # Fix: str(NaN) == "nan" was being counted as a real answer
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "-"}


def normalize(text: str) -> str:
    text = strip_accents(str(text).lower())
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


SPANISH_STOPWORDS = sorted({strip_accents(w) for w in """
a al algo algun alguna algunas alguno algunos ante antes aqui asi aun cada como con
contra cual cuales cuando de del desde donde dos durante e el ella ellas ellos en
entre era eran eres es esa esas ese eso esos esta estaba estado estan estar estas este
esto estos estoy fue fueron ha habia han has hasta hay he la las le les lo los me mi
mis mucho muchos muy nos nosotros o os otra otras otro otros para pero poco por porque
pues que quien se sea ser si sido sobre solo son soy su sus tambien te ti tiene tienen
todo todos tu tus un una uno unos usted ustedes vez y ya yo jaja jajaja
""".split()})

# =========================================================
# SENTIMENT — Spanish lexicon (fast, no downloads)
# VADER only understands English, so it scored Spanish text as ~neutral.
# =========================================================
def _lex(words, w):
    return {strip_accents(x): w for x in words.split()}

LEXICON = {}
LEXICON.update(_lex("""excelente increible genial perfecto perfecta maravilloso maravillosa
encanta encanto encantan encantaron buenisimo buenisima espectacular impresionante
fantastico fantastica golazo partidazo brutal""", 2.0))
LEXICON.update(_lex("""bueno buena buenos buenas bien gusta gusto gustan recomendado recomendada
recomiendo recomendable rapido rapida rapidos facil util utiles feliz contento contenta
gracias top mejor mejores amable amables eficiente satisfecho satisfecha funciona funcionan
sirve merecido orgulloso orgullosa chido chida padre felicidades bonito bonita lindo linda
agradezco agradecido resolvieron solucionaron ayudo ayudaron atino confirmo""", 1.0))
LEXICON.update(_lex("""pesimo pesima horrible terrible decepcion decepcionado decepcionada
decepcionante basura fatal estafa odio asco verguenza robo robaron""", -2.0))
LEXICON.update(_lex("""malo mala malos malas mal lento lenta lentos lentisimo lentisima tarda
tardan demora demoran error errores falla fallas fallo bug bugs problema problemas caro caros
carisimo carisima peor peores enganosa enganoso absurdo absurda molesto molesta enojado
enojada queja quejas reclamo mentira inutil cancelaron cancelado caida traba calienta rayada
rayado eliminados roban pena lloran triste lamentable arrepiento dificil complicado empeoro
empeorado""", -1.0))
LEXICON.update(_lex("esperando nadie cobraron exageras anuncios", -0.5))
LEXICON.update(_lex("buen buenazo triunfo completo salvo carisimos carisimas invento inventa", 0.0))
LEXICON.update({"buen": 1.0, "buenazo": 1.5, "triunfo": 1.0, "completo": 0.5, "salvo": 1.0,
                "carisimos": -1.0, "carisimas": -1.0, "invento": -1.0, "inventa": -1.0})

# Multi-word expressions, matched before tokenizing (accent-free, lowercase)
PHRASES = {
    "vale la pena": 0.0, "como nunca": 1.0, "otro nivel": 1.5, "on fire": 1.5,
    "ya no se puede": -1.5, "no puede ser": -1.0, "se acaba": -1.0, "no sabe": -1.0,
    "nunca responde": -1.5, "nunca contesta": -1.5, "nadie contesta": -1.5,
    "nadie responde": -1.5, "no contestan": -1.5, "no responden": -1.5, "no corre": -1.0,
    "sin rodeos": 0.5, "ya era hora": 0.5, "nada que ver": -0.5,
}

NEGATORS = {"no", "ni", "nunca", "jamas", "sin", "tampoco"}
INTENSIFIERS = {"muy", "super", "demasiado", "bastante", "tan", "totalmente",
                "extremadamente", "muchisimo", "re"}
EMOJI_POS = set("😀😃😄😁😊😍🥰😎👍👏🔥💯💪❤♥🙌✨🎉👌🤩🥳")
EMOJI_NEG = set("😡😠🤬😤😞😢😭👎💩🙄😒😫😩🤮")
TOKEN_RE = re.compile(r"[a-zñ]+")


def lexicon_score(text: str) -> float:
    raw = str(text)
    score = 1.5 * sum(ch in EMOJI_POS for ch in raw) - 1.5 * sum(ch in EMOJI_NEG for ch in raw)
    low = " " + " ".join(TOKEN_RE.findall(strip_accents(raw.lower()))) + " "
    for ph, w in PHRASES.items():
        hits = low.count(f" {ph} ")
        if hits:
            score += w * hits
            low = low.replace(f" {ph} ", " ")
    toks = low.split()
    for i, tok in enumerate(toks):
        w = LEXICON.get(tok)
        if w is None:
            continue
        if i > 0 and toks[i - 1] in INTENSIFIERS:
            w *= 1.3
        if any(t in NEGATORS for t in toks[max(0, i - 3):i]):
            w *= -0.74
        score += w
    return float(score / np.sqrt(score ** 2 + 15))  # VADER-style normalization to [-1, 1]


def to_label(s: float, thr: float = 0.05) -> str:
    return "Positive" if s >= thr else ("Negative" if s <= -thr else "Neutral")


@st.cache_resource(show_spinner="Loading pysentimiento model…")
def _pysent():
    from pysentimiento import create_analyzer
    return create_analyzer(task="sentiment", lang="es")


@st.cache_data(show_spinner="Scoring sentiment…")
def score_texts(texts: tuple, method: str) -> np.ndarray:
    if method.startswith("pysentimiento"):
        an = _pysent()
        preds = an.predict(list(texts))
        return np.array([p.probas.get("POS", 0) - p.probas.get("NEG", 0) for p in preds])
    return np.array([lexicon_score(t) for t in texts])

# =========================================================
# STATISTICS HELPERS
# =========================================================
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


def mean_ci(x: pd.Series):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return (x.mean() if n else np.nan, np.nan, np.nan)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return (m, m - h, m + h)


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for lab in LABELS:
        k = int((df["sentiment"] == lab).sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"sentiment": lab, "n": k, "%": 100 * k / n if n else 0,
                     "95% CI low": 100 * lo, "95% CI high": 100 * hi})
    return pd.DataFrame(rows)


def cramers_v(ct: pd.DataFrame, chi2: float) -> float:
    n = ct.values.sum()
    r, c = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else np.nan


def group_stats(df: pd.DataFrame, col: str, min_n: int = 1):
    g = df.groupby(col, dropna=False)
    out = []
    for name, d in g:
        m, lo, hi = mean_ci(d["sent_score"])
        out.append({col: name, "n": len(d), "share %": 100 * len(d) / len(df),
                    "mean score": m, "CI low": lo, "CI high": hi,
                    "% positive": 100 * (d["sentiment"] == "Positive").mean(),
                    "% negative": 100 * (d["sentiment"] == "Negative").mean()})
    table = pd.DataFrame(out).query("n >= @min_n").sort_values("n", ascending=False)

    tests = {}
    sub = df[df[col].isin(table[col])]
    ct = pd.crosstab(sub[col], sub["sentiment"])
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        tests["chi2"] = (chi2, p, cramers_v(ct, chi2), (exp < 5).mean() * 100)
    samples = [d["sent_score"].values for _, d in sub.groupby(col) if len(d) > 1]
    if len(samples) > 1:
        tests["kruskal"] = stats.kruskal(*samples)
    return table, tests


def describe_tests(tests: dict) -> str:
    lines = []
    if "chi2" in tests:
        chi2, p, v, low_exp = tests["chi2"]
        sig = "significant" if p < 0.05 else "not significant"
        strength = "weak" if v < 0.1 else "moderate" if v < 0.3 else "strong"
        lines.append(f"**Chi-square (label distribution):** χ²={chi2:.1f}, p={p:.4f} → {sig}; "
                     f"Cramér's V={v:.2f} ({strength} association)")
        if low_exp > 20:
            lines.append(f"⚠️ {low_exp:.0f}% of expected cells < 5 — chi-square may be unreliable.")
    if "kruskal" in tests:
        h, p = tests["kruskal"]
        sig = "significant" if p < 0.05 else "not significant"
        lines.append(f"**Kruskal-Wallis (score):** H={h:.1f}, p={p:.4f} → {sig}")
    return "  \n".join(lines) if lines else "Not enough groups for statistical tests."


def distinctive_terms(texts_a, texts_b, top_n=15):
    """Log-odds ratio (add-1 smoothing) of terms in group A vs group B."""
    if len(texts_a) == 0 or len(texts_b) == 0:
        return pd.DataFrame()
    cv = CountVectorizer(stop_words=SPANISH_STOPWORDS, ngram_range=(1, 2), min_df=2)
    try:
        X = cv.fit_transform(list(texts_a) + list(texts_b))
    except ValueError:
        return pd.DataFrame()
    na = len(texts_a)
    a = np.asarray(X[:na].sum(axis=0)).ravel() + 1
    b = np.asarray(X[na:].sum(axis=0)).ravel() + 1
    lor = np.log(a / a.sum()) - np.log(b / b.sum())
    idx = lor.argsort()[::-1][:top_n]
    terms = cv.get_feature_names_out()
    return pd.DataFrame({"term": terms[idx], "count": a[idx] - 1, "log-odds": lor[idx].round(2)})

# =========================================================
# TOPICS
# =========================================================
DEFAULT_RULES = """Payment & Billing: pag, factura, cobr, reembols, tarjeta, monto, referencia, dinero, precio
Response Time: lent, tard, espera, demor, rapid, minutos, tiempo
Human Agent Request: humano, asesor, agente, persona, transfer, call center, llamar
Technical Errors: error, falla, bug, traba, se cierra, cerro, reinstal, descarg, pdf, no funciona
Account & Login: contrasena, password, acceso, acceder, entrar, cuenta, codigo, verificacion, bloque, correo
Delivery & Orders: pedido, envio, entrega, guia, rastre, direccion, llega"""


def parse_rules(text: str) -> dict:
    rules = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, terms = line.split(":", 1)
        terms = [normalize(t) for t in terms.split(",") if normalize(t)]
        if name.strip() and terms:
            rules[name.strip()] = terms
    return rules


def rule_topics(texts_norm, rules: dict) -> np.ndarray:
    """Assign each text to the rule with most term hits (terms match word prefixes)."""
    pats = {n: [re.compile(r"\b" + re.escape(t)) for t in ts] for n, ts in rules.items()}
    out = []
    for s in texts_norm:
        hits = {n: sum(bool(p.search(s)) for p in ps) for n, ps in pats.items()}
        best = max(hits, key=hits.get) if hits else None
        out.append(best if best and hits[best] > 0 else None)
    return np.array(out, dtype=object)


@st.cache_data(show_spinner="Detecting topics…")
def fit_topics(texts_norm: tuple, k: int | None, max_features: int, seed: int = 42):
    n = len(texts_norm)
    tfidf = TfidfVectorizer(max_features=max_features, stop_words=SPANISH_STOPWORDS,
                            ngram_range=(1, 2), min_df=2 if n >= 50 else 1, sublinear_tf=True)
    X = tfidf.fit_transform(texts_norm)
    n_unique = len(set(texts_norm))
    k_max = max(2, min(10, n_unique - 1))
    rng = np.random.RandomState(seed)
    sample = rng.choice(n, min(n, 3000), replace=False)

    def sil(lbl):
        return silhouette_score(X[sample], lbl[sample]) if len(set(lbl[sample])) > 1 else np.nan

    sil_by_k = {}
    if k is None:
        for kk in range(2, k_max + 1):
            lbl = KMeans(n_clusters=kk, random_state=seed, n_init=10).fit_predict(X)
            sil_by_k[kk] = sil(lbl)
        best = np.nanmax(list(sil_by_k.values()))
        # smallest k within 90% of the best silhouette (avoids over-splitting short texts)
        k = min(kk for kk, v in sil_by_k.items() if np.nan_to_num(v, nan=-1) >= 0.9 * best)
    k = min(k, k_max)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    return X, labels, tfidf.get_feature_names_out(), sil(labels), sil_by_k


def keyword_names(X, labels, feature_names, top_n=5, prefix=""):
    names, kw_map = {}, {}
    used = set()
    for t in np.unique(labels):
        idx = np.where(labels == t)[0]
        mean = np.asarray(X[idx].mean(axis=0)).ravel()
        kws = [feature_names[i] for i in mean.argsort()[::-1][:top_n]]
        kw_map[t] = ", ".join(kws)
        name = prefix + " / ".join(kws[:3]).title()
        if name in used:
            name = f"{name} ({kws[3] if len(kws) > 3 else t})"
        used.add(name)
        names[t] = name
    return names, kw_map


def assign_topics(texts_norm, mode, rules, k, max_features):
    """Returns topic_name array, keywords array, info dict."""
    texts_norm = np.asarray(texts_norm, dtype=object)
    names = np.empty(len(texts_norm), dtype=object)
    kws = np.empty(len(texts_norm), dtype=object)
    info = {"sil": np.nan, "sil_by_k": {}, "rule_coverage": 0.0}
    if mode.startswith("Rules") and rules:
        r = rule_topics(texts_norm, rules)
        mask = np.array([x is not None for x in r])
        names[mask] = r[mask]
        for n_, ts in rules.items():
            kws[names == n_] = ", ".join(ts[:6])
        info["rule_coverage"] = mask.mean()
        rest = np.where(~mask)[0]
        prefix = "Other: "
    else:
        rest = np.arange(len(texts_norm))
        prefix = ""
    if len(rest) >= 6 and len(set(texts_norm[rest])) >= 3:
        X, lbl, feats, sil, sil_by_k = fit_topics(tuple(texts_norm[rest]), k, max_features)
        nm, kw = keyword_names(X, lbl, feats, prefix=prefix)
        names[rest] = [nm[t] for t in lbl]
        kws[rest] = [kw[t] for t in lbl]
        info.update(sil=sil, sil_by_k=sil_by_k)
    elif len(rest):
        names[rest] = "Other"
        kws[rest] = ""
    return names, kws, info


# =========================================================
# DATA PREP
# =========================================================
@st.cache_data
def load_csv(data: bytes, sep: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            if sep == "auto":
                return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
            return pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file (tried UTF-8 and Latin-1).")


def detect_response_columns(df):
    cols = [c for c in df.columns if str(c).lower().startswith("respuesta")]
    return sorted(cols, key=lambda c: int(re.findall(r"\d+", str(c))[0]) if re.findall(r"\d+", str(c)) else 0)


def position_of(col: str):
    m = re.findall(r"\d+", str(col))
    return int(m[0]) if m else np.nan


def detect_date_column(df):
    for c in df.columns:
        if any(k in str(c).lower() for k in ("fecha", "date", "time", "timestamp")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return c
    return None


def categorical_columns(df, exclude):
    out = []
    for c in df.columns:
        if c in exclude or pd.api.types.is_numeric_dtype(df[c]):
            continue
        nu = df[c].nunique(dropna=True)
        if 2 <= nu <= 30 and nu / max(len(df), 1) < 0.5:
            out.append(c)
    return out


def numeric_columns(df, exclude):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            and not re.search(r"(^|_)id$|^id", str(c).lower()) and df[c].nunique() > 2]


def to_long(df_raw: pd.DataFrame, text_cols: list, unit: str) -> pd.DataFrame:
    base = df_raw.reset_index(drop=False).rename(columns={"index": "_row"})
    if unit == "Each response":
        long = base.melt(id_vars=[c for c in base.columns if c not in text_cols],
                         value_vars=text_cols, var_name="source_column", value_name="text")
        long = long[~long["text"].map(is_empty)].copy()
        long["position"] = long["source_column"].map(position_of)
    else:
        long = base.copy()
        long["text"] = base[text_cols].apply(
            lambda r: " ".join(str(v).strip() for v in r if not is_empty(v)), axis=1)
        long["source_column"] = "+".join(text_cols)
        long["position"] = np.nan
        long = long[long["text"].str.len() > 0]
    long["text"] = long["text"].astype(str).str.strip()
    long["n_words"] = long["text"].str.split().str.len()
    return long.reset_index(drop=True)


LABEL_MAP = {"positivo": "Positive", "positive": "Positive", "pos": "Positive", "1": "Positive",
             "negativo": "Negative", "negative": "Negative", "neg": "Negative", "-1": "Negative",
             "neutral": "Neutral", "neutro": "Neutral", "neu": "Neutral", "0": "Neutral"}

# =========================================================
# UI
# =========================================================
def setup_page():
    st.set_page_config(page_title="Survey Bot Insights", layout="wide")
    st.markdown("""
    <style>
    :root { --bg:#FFFFFF; --panel:#FFFFFF; --text:#3A3A3A; --muted:#6B6B6B;
            --line:#E3E3E3; --accent:#F2C300; }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg) !important; color: var(--text) !important;
        font-family: "Avenir Next", "Avenir", "Segoe UI", system-ui, sans-serif; }
    /* every text element: force readable gray, whatever the browser theme */
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp small, .stApp strong { color: var(--text); }
    h1 { font-size: 2.3rem !important; font-weight: 600 !important; color: #4A4A4A !important; }
    h2, h3 { font-weight: 500 !important; color: #4A4A4A !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

    /* sidebar */
    [data-testid="stSidebar"] { background-color: #F7F7F7 !important; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 500; opacity: 1 !important; }
    [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }

    /* inputs */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea, input {
        background-color: #FFFFFF !important; color: var(--text) !important; border-color: #CFCFCF !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #FFFFFF !important; border: 1px dashed #BDBDBD !important; }

    /* tabs */
    button[data-baseweb="tab"] p { color: var(--muted) !important; font-size: 1rem; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #3A3A3A !important; font-weight: 600; }
    [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
    [data-baseweb="tab-border"] { background-color: var(--line) !important; }

    /* metric cards */
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line);
        border-top: 3px solid var(--accent); border-radius: 12px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .9rem !important; }
    [data-testid="stMetricValue"] { color: #3A3A3A !important; font-size: 1.7rem !important; }
    [data-testid="stMetricValue"] div { overflow: visible !important; text-overflow: clip !important; }

    /* tables, alerts, dividers */
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
    [data-testid="stAlert"] { background-color: #FFFBEA !important; border: 1px solid #F2C300 !important; }
    [data-testid="stAlert"] * { color: var(--text) !important; }
    hr { border: none; border-top: 1px solid var(--line) !important; margin: 1.5rem 0; }
    .stDownloadButton button, .stButton button { background:#FFFFFF !important; color: var(--text) !important;
        border: 1px solid var(--accent) !important; }
    </style>""", unsafe_allow_html=True)
    st.title("Survey Bot Insights")
    st.caption("Understand what users are really saying and where your bot needs improvement")
    st.divider()


def fmt_pct(x):
    return f"{x:.1f}%"


def executive_summary(df, topic_tbl, seg_findings):
    total, rows = len(df), df["_row"].nunique()
    dist = sentiment_distribution(df).set_index("sentiment")
    neg = dist.loc["Negative"]
    m, lo, hi = mean_ci(df["sent_score"])
    txt = [f"**Overview** — {total:,} texts analyzed from {rows:,} records.",
           f"- Negative: **{neg['%']:.1f}%** (95% CI {neg['95% CI low']:.1f}–{neg['95% CI high']:.1f}%) · "
           f"Positive: **{dist.loc['Positive','%']:.1f}%** · Neutral: **{dist.loc['Neutral','%']:.1f}%**",
           f"- Mean sentiment score: **{m:.2f}** (95% CI {lo:.2f} to {hi:.2f}, scale −1 to 1)",
           "", "**Most frequent topics**"]
    for _, r in topic_tbl.head(3).iterrows():
        txt.append(f"- **{r['topic_name']}** — {r['n']} texts ({r['share %']:.1f}%), {r['% negative']:.0f}% negative")
    worst = topic_tbl[topic_tbl["n"] >= 5].sort_values("mean score").head(2)
    if not worst.empty:
        txt += ["", "**Topics generating most frustration**"]
        for _, r in worst.iterrows():
            txt.append(f"- **{r['topic_name']}** — mean score {r['mean score']:.2f}, {r['% negative']:.0f}% negative")
    if seg_findings:
        txt += ["", "**Statistically significant differences (p < 0.05)**"] + [f"- {s}" for s in seg_findings]
    txt += ["", "**Recommendation** — prioritize the high-volume, high-negativity topics for new intents, "
            "better answers, or escalation to human agents."]
    return "\n".join(txt)


def main(df_override: pd.DataFrame | None = None):
    setup_page()

    with st.sidebar:
        st.subheader("Data")
        uploaded = st.file_uploader("Upload your CSV", type=["csv"])
        sep = st.selectbox("CSV separator", ["auto", ",", ";", "|", "\t"], index=0)

    if df_override is not None:
        df_raw = df_override
    elif uploaded:
        try:
            df_raw = load_csv(uploaded.getvalue(), sep)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            st.stop()
    else:
        st.info("Upload a CSV with columns like respuesta1, respuesta2, …")
        st.stop()

    resp_default = detect_response_columns(df_raw)
    text_candidates = [c for c in df_raw.columns if not pd.api.types.is_numeric_dtype(df_raw[c])
                       and not pd.api.types.is_datetime64_any_dtype(df_raw[c])]
    text_candidates = list(dict.fromkeys(resp_default + text_candidates))

    with st.sidebar:
        text_cols = st.multiselect("Text columns to analyze", text_candidates,
                                   default=resp_default or text_candidates[:1])
        unit = st.radio("Unit of analysis", ["Each response", "Each record (texts combined)"],
                        help="'Each response' counts every answer separately — recommended for "
                             "statistics. 'Each record' joins all selected columns per row.")
        st.divider()
        st.subheader("Sentiment")
        methods = ["Spanish lexicon (fast)"]
        if HAS_PYSENTIMIENTO:
            methods.append("pysentimiento (accurate, slower)")
        method = st.selectbox("Method", methods)
        thr = st.slider("Neutral band (±)", 0.0, 0.5, 0.05, 0.01)
        st.divider()
        st.subheader("Topic detection")
        topic_mode = st.radio("Method", ["Rules + clustering (recommended)", "Clustering only"],
                              help="Rules assign known topics by keywords; texts that match no rule "
                                   "are grouped automatically.")
        rules_text = st.text_area("Topic rules (Topic: term1, term2, …)", DEFAULT_RULES, height=220,
                                  disabled=topic_mode == "Clustering only",
                                  help="Terms match the start of words: 'rastre' matches rastreo/rastrear.")
        auto_k = st.toggle("Choose number of clusters automatically", value=True)
        k_manual = st.slider("Number of clusters", 2, 10, 5, disabled=auto_k)
        max_features = st.slider("TF-IDF features", 500, 20000, 8000, step=500)
        min_n = st.number_input("Min. texts per group to report", 1, 1000, 5)

    if not text_cols:
        st.error("Select at least one text column (e.g. respuesta1, respuesta2, …).")
        st.stop()

    df = to_long(df_raw, text_cols, unit)
    if len(df) < 3:
        st.error("Fewer than 3 non-empty texts — not enough to analyze.")
        st.stop()

    df["sent_score"] = score_texts(tuple(df["text"]), method)
    df["sentiment"] = df["sent_score"].map(lambda s: to_label(s, thr))
    df["text_norm"] = df["text"].map(normalize)

    try:
        tn, tk, tinfo = assign_topics(df["text_norm"].values, topic_mode, parse_rules(rules_text),
                                      None if auto_k else int(k_manual), int(max_features))
    except ValueError:
        st.error("Texts are too short or only contain stopwords — topics cannot be computed.")
        st.stop()
    df["topic_name"] = tn
    df["topic_keywords"] = tk
    sil, sil_by_k = tinfo["sil"], tinfo["sil_by_k"]

    exclude = set(text_cols) | {"_row", "text", "text_norm", "source_column", "position", "n_words",
                                "sent_score", "sentiment", "topic_name", "topic_keywords"}
    date_col = detect_date_column(df_raw)
    label_cols = {c for c in df_raw.columns if df_raw[c].astype(str).str.lower().str.strip()
                  .isin(LABEL_MAP).mean() > 0.8}
    cat_cols = categorical_columns(df_raw, exclude | label_cols | ({date_col} if date_col else set()))
    num_cols = numeric_columns(df_raw, exclude)

    topic_tbl, topic_tests = group_stats(df, "topic_name")
    kw_by_name = df.drop_duplicates("topic_name").set_index("topic_name")["topic_keywords"]
    topic_tbl["keywords"] = topic_tbl["topic_name"].map(kw_by_name)

    seg_findings = []
    seg_results = {}
    for c in cat_cols:
        tbl, tests = group_stats(df, c, min_n)
        seg_results[c] = (tbl, tests)
        if "chi2" in tests and tests["chi2"][1] < 0.05 and len(tbl) > 1:
            top_neg = tbl.sort_values("% negative", ascending=False).iloc[0]
            seg_findings.append(f"Sentiment differs by **{c}** (p={tests['chi2'][1]:.3f}); "
                                f"most negative: **{top_neg[c]}** ({top_neg['% negative']:.0f}% negative)")

    tabs = st.tabs(["Overview", "Sentiment", "Topics", "Segments", "Language", "Validation", "Data"])

    # ---------------- Overview ----------------
    with tabs[0]:
        rows = df["_row"].nunique()
        c = st.columns(5)
        c[0].metric("Records", f"{len(df_raw):,}")
        c[1].metric("Texts analyzed", f"{len(df):,}")
        c[2].metric("Avg texts / record", f"{len(df) / max(len(df_raw), 1):.2f}")
        c[3].metric("Negative", fmt_pct((df["sentiment"] == "Negative").mean() * 100))
        m, lo, hi = mean_ci(df["sent_score"])
        c[4].metric("Mean score", f"{m:.2f}", help=f"95% CI {lo:.2f} to {hi:.2f}")
        st.caption(f"{len(df_raw) - rows} records had no usable text and were excluded.")

        st.markdown(executive_summary(df, topic_tbl, seg_findings))

        if unit == "Each response" and len(text_cols) > 1:
            st.subheader("Response completion by column")
            fill = pd.DataFrame({"column": text_cols,
                                 "filled": [int((~df_raw[c].map(is_empty)).sum()) for c in text_cols]})
            fill["fill rate %"] = 100 * fill["filled"] / len(df_raw)
            st.bar_chart(fill.set_index("column")["fill rate %"], height=260)
            counts = df_raw[text_cols].apply(lambda r: sum(not is_empty(v) for v in r), axis=1)
            dist = counts.value_counts().sort_index().rename_axis("responses per record").rename("records")
            st.markdown(f"Responses per record — mean **{counts.mean():.2f}**, median **{counts.median():.0f}**, "
                        f"std **{counts.std():.2f}**, records with 0: **{(counts == 0).sum()}**")
            st.bar_chart(dist, height=220)

    # ---------------- Sentiment ----------------
    with tabs[1]:
        st.subheader("Sentiment distribution (95% Wilson CI)")
        d = sentiment_distribution(df)
        st.dataframe(d.round(1), hide_index=True, width="stretch")
        st.bar_chart(d.set_index("sentiment")["%"], height=260)

        st.subheader("Score statistics")
        s = df["sent_score"]
        st.dataframe(pd.DataFrame({"stat": ["mean", "median", "std", "min", "p25", "p75", "max", "skew"],
                                   "value": [s.mean(), s.median(), s.std(), s.min(), s.quantile(.25),
                                             s.quantile(.75), s.max(), stats.skew(s)]}).round(3),
                     hide_index=True)
        hist = pd.cut(s, bins=np.linspace(-1, 1, 21)).value_counts().sort_index()
        hist.index = [f"{i.left:.1f}" for i in hist.index]
        st.bar_chart(hist, height=240)

        if unit == "Each response" and df["position"].notna().any():
            st.subheader("Sentiment by response position")
            pos_tbl, pos_tests = group_stats(df, "position", 1)
            st.dataframe(pos_tbl.sort_values("position").round(2), hide_index=True, width="stretch")
            st.markdown(describe_tests(pos_tests))
            rho, p = stats.spearmanr(df["position"], df["sent_score"])
            st.caption(f"Spearman ρ (position vs score) = {rho:.3f}, p = {p:.4f}")

        if date_col:
            st.subheader(f"Trend over time ({date_col})")
            dd = df.assign(_date=pd.to_datetime(df[date_col], errors="coerce")).dropna(subset=["_date"])
            span = (dd["_date"].max() - dd["_date"].min()).days
            freq = "D" if span <= 31 else "W" if span <= 365 else "MS"
            tr = dd.set_index("_date").resample(freq).agg(
                texts=("text", "count"), mean_score=("sent_score", "mean"),
                pct_negative=("sentiment", lambda x: 100 * (x == "Negative").mean()))
            st.line_chart(tr[["mean_score"]], height=240, color="#F2C300")
            st.line_chart(tr[["pct_negative"]], height=240, color="#7A7A7A")
            st.dataframe(tr.round(2), width="stretch")

    # ---------------- Topics ----------------
    with tabs[2]:
        st.subheader("Topics")
        parts = [f"{df['topic_name'].nunique()} topics"]
        if topic_mode.startswith("Rules"):
            parts.append(f"{100 * tinfo['rule_coverage']:.1f}% of texts matched a rule")
        if not np.isnan(sil):
            q = "good" if sil > 0.25 else "fair" if sil > 0.1 else "weak"
            parts.append(f"clusters silhouette = {sil:.3f} ({q} separation)")
        st.caption(" · ".join(parts))
        if sil_by_k:
            st.caption("Cluster quality (silhouette) by number of clusters")
            st.line_chart(pd.Series(sil_by_k, name="silhouette").rename_axis("k"), height=200, color="#7A7A7A")
        st.dataframe(topic_tbl.round(2), hide_index=True, width="stretch")
        st.bar_chart(topic_tbl.set_index("topic_name")[["% positive", "% negative"]], height=320, color=["#F2C300", "#7A7A7A"])
        st.markdown(describe_tests(topic_tests))

        st.subheader("Suggested new intents / pain points")
        pain = topic_tbl[(topic_tbl["n"] >= min_n) &
                         ((topic_tbl["mean score"] < -0.1) | (topic_tbl["% negative"] >= 40))]
        if pain.empty:
            st.success("No critical pain points detected.")
        else:
            for _, r in pain.sort_values("mean score").iterrows():
                st.markdown(f"- **{r['topic_name']}** → {r['n']} texts, {r['% negative']:.0f}% negative, "
                            f"mean score {r['mean score']:.2f} · _{r['keywords']}_")

        st.subheader("Examples")
        chosen = st.selectbox("Select topic", sorted(df["topic_name"].unique()))
        sub = df[df["topic_name"] == chosen]
        st.dataframe(sub[["text", "sentiment", "sent_score"]].sort_values("sent_score"),
                     hide_index=True, width="stretch")

    # ---------------- Segments ----------------
    with tabs[3]:
        if not cat_cols:
            st.info("No categorical columns found (e.g. platform, channel, topic) to segment by.")
        else:
            col = st.selectbox("Segment by", cat_cols)
            tbl, tests = seg_results[col]
            st.dataframe(tbl.round(2), hide_index=True, width="stretch")
            st.markdown(describe_tests(tests))
            st.bar_chart(tbl.set_index(col)[["% positive", "% negative"]], height=300, color=["#F2C300", "#7A7A7A"])
            st.subheader(f"Topics × {col} (% of texts within each {col})")
            st.dataframe((pd.crosstab(df["topic_name"], df[col], normalize="columns") * 100).round(1),
                         width="stretch")
        if num_cols:
            st.subheader("Engagement vs sentiment (record level)")
            rec = df.groupby("_row")["sent_score"].mean().rename("mean_score").to_frame()
            rec = rec.join(df_raw[num_cols])
            res = []
            for c in num_cols:
                ok = rec[[c, "mean_score"]].dropna()
                if len(ok) > 3:
                    rho, p = stats.spearmanr(ok[c], ok["mean_score"])
                    res.append({"variable": c, "Spearman ρ": rho, "p-value": p, "n": len(ok)})
            if res:
                st.dataframe(pd.DataFrame(res).round(4), hide_index=True)

    # ---------------- Language ----------------
    with tabs[4]:
        st.subheader("Text length")
        st.dataframe(df.groupby("sentiment")["n_words"].describe().round(1), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distinctive terms — negative texts**")
            st.dataframe(distinctive_terms(df.loc[df.sentiment == "Negative", "text_norm"],
                                           df.loc[df.sentiment != "Negative", "text_norm"]),
                         hide_index=True, width="stretch")
        with c2:
            st.markdown("**Distinctive terms — positive texts**")
            st.dataframe(distinctive_terms(df.loc[df.sentiment == "Positive", "text_norm"],
                                           df.loc[df.sentiment != "Positive", "text_norm"]),
                         hide_index=True, width="stretch")
        st.subheader("Most repeated texts")
        dup = df["text"].value_counts()
        st.caption(f"{dup.size:,} unique texts out of {len(df):,} ({100 * dup.size / len(df):.1f}% unique)")
        st.dataframe(dup.head(15).rename_axis("text").rename("count").reset_index(),
                     hide_index=True, width="stretch")

    # ---------------- Validation ----------------
    with tabs[5]:
        st.markdown("Compare the model against a column with human/known sentiment labels.")
        label_candidates = [c for c in df_raw.columns if df_raw[c].astype(str).str.lower()
                            .str.strip().isin(LABEL_MAP).mean() > 0.8]
        if not label_candidates:
            st.info("No label column found (values like positivo / negativo / neutral).")
        else:
            lc = st.selectbox("Label column", label_candidates)
            others = [c for c in text_candidates if c != lc and c not in (date_col,)]
            true_all = df_raw[lc].astype(str).str.lower().str.strip().map(LABEL_MAP)

            def agreement(col):
                ok = ~df_raw[col].map(is_empty) & true_all.notna()
                if ok.sum() == 0:
                    return 0.0
                pr = [to_label(x, thr) for x in score_texts(tuple(df_raw.loc[ok, col].astype(str)), method)]
                return float((np.array(pr) == true_all[ok].values).mean())

            scores = {c: agreement(c) for c in others}
            default_ix = others.index(max(scores, key=scores.get)) if scores else 0
            tc = st.selectbox("Text column the label refers to", others, index=default_ix,
                              help="Pre-selected: the column whose text agrees most with the labels.")
            v = df_raw[[tc, lc]].copy()
            v = v[~v[tc].map(is_empty)]
            v["true"] = v[lc].astype(str).str.lower().str.strip().map(LABEL_MAP)
            v = v.dropna(subset=["true"])
            v["pred"] = [to_label(s, thr) for s in score_texts(tuple(v[tc].astype(str)), method)]
            c = st.columns(3)
            c[0].metric("Accuracy", fmt_pct(100 * accuracy_score(v["true"], v["pred"])))
            c[1].metric("Macro F1", f"{f1_score(v['true'], v['pred'], average='macro'):.3f}")
            c[2].metric("Cohen's κ", f"{cohen_kappa_score(v['true'], v['pred']):.3f}")
            cm = pd.DataFrame(confusion_matrix(v["true"], v["pred"], labels=LABELS),
                              index=[f"true {l}" for l in LABELS], columns=[f"pred {l}" for l in LABELS])
            st.dataframe(cm, width="stretch")
            per = f1_score(v["true"], v["pred"], labels=LABELS, average=None)
            st.dataframe(pd.DataFrame({"class": LABELS, "F1": per.round(3)}), hide_index=True)
            st.markdown("**Misclassified examples**")
            st.dataframe(v[v["true"] != v["pred"]][[tc, "true", "pred"]].head(30),
                         hide_index=True, width="stretch")

    # ---------------- Data / Export ----------------
    with tabs[6]:
        out_cols = ["_row", "source_column", "text", "topic_name", "topic_keywords", "sentiment",
                    "sent_score"] + [c for c in cat_cols + num_cols + ([date_col] if date_col else [])]
        out = df[[c for c in dict.fromkeys(out_cols) if c in df.columns]]
        st.dataframe(out.head(500), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("Download results CSV", out.to_csv(index=False).encode("utf-8-sig"),
                           "survey_bot_insights_results.csv", "text/csv")
        c2.download_button("Download topic summary CSV",
                           topic_tbl.round(3).to_csv(index=False).encode("utf-8-sig"),
                           "survey_bot_insights_topics.csv", "text/csv")


if __name__ == "__main__":
    main()