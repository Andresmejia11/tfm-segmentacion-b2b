import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from minisom import MiniSom
import warnings
warnings.filterwarnings("ignore")
# ── Carga de datos desde GitHub ────────────────────────────
@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos():
    base = "https://raw.githubusercontent.com/Andresmejia11/tfm-segmentacion-b2b/main/"
    
    clientes = pd.read_csv(base + "CLIENTES.txt", sep="|", encoding="latin-1")
    ventas   = pd.read_csv(base + "VENTAS.txt",   sep="|", encoding="latin-1")
    
    import zipfile, io, requests
    r = requests.get(base + "CONSULTAS.zip")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        nombre = [f for f in z.namelist() if f.endswith(".txt")][0]
        with z.open(nombre) as f:
            consultas = pd.read_csv(f, sep="|", encoding="latin-1")

    return clientes, ventas, consultas

clientes, ventas, consultas = cargar_datos()
# ── Configuración ──────────────────────────────────────────
st.set_page_config(
    page_title="Segmentación Clientes B2B · Colombia",
    page_icon="📊",
    layout="wide"
)

# ── Menú lateral ───────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Navegación")

seccion = st.sidebar.radio("", [
    "🏠 Inicio",
    "📊 Segmentación",
    "🔮 Predicción",
    "⚖️ Comparación"
])

# ── INICIO ─────────────────────────────────────────────────
if seccion == "🏠 Inicio":
    st.title("📊 Segmentación de Clientes B2B")
    st.subheader("Análisis de recurrencia en el sector de información empresarial · Colombia")
    st.markdown("---")

    # Procesamiento básico
    ventas_agg = ventas.groupby("ID").agg(
        TOTAL_VENTAS=("IMPORTE", "sum"),
        PROMEDIO_VENTA=("IMPORTE", "mean"),
    ).reset_index()

    consultas_agg = consultas.groupby("ID").agg(
        NUM_CONSULTAS=("IDCONSUMO", "count")
    ).reset_index()

    df = clientes.merge(ventas_agg, on="ID", how="left").merge(consultas_agg, on="ID", how="left")
    df["NUM_CONSULTAS"] = df["NUM_CONSULTAS"].fillna(0).astype(int)

    naturales = ["PERSONA FISICA", "EMPRESARIO"]
    df["TIPO_CLIENTE"] = df["FORMAJURIDICA"].apply(
        lambda x: "NATURAL" if x in naturales else "JURIDICO"
    )

    total     = len(df)
    n_nat     = (df["TIPO_CLIENTE"] == "NATURAL").sum()
    n_jur     = (df["TIPO_CLIENTE"] == "JURIDICO").sum()
    pct_nat   = n_nat / total * 100
    pct_jur   = n_jur / total * 100

    # KPIs
    st.markdown("### Resumen general")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total clientes", f"{total:,}")
    k2.metric("Clientes Naturales", f"{n_nat:,}", f"{pct_nat:.1f}%")
    k3.metric("Clientes Jurídicos", f"{n_jur:,}", f"{pct_jur:.1f}%")
    k4.metric("Variables analizadas", "4")

    st.markdown("---")
    st.markdown("### ¿Qué encontramos?")
    col1, col2 = st.columns(2)
    with col1:
        st.success("🔵 **Clientes Naturales** · 3 segmentos identificados: Ocasionales, Recurrentes e Intensivos")
        st.success("🟣 **Clientes Jurídicos** · 3 segmentos identificados: Ocasionales, Recurrentes e Intensivos")
    with col2:
        st.info("🎯 **Predicción Naturales** · Random Forest 96% de precisión")
        st.info("🎯 **Predicción Jurídicos** · Random Forest 89% de precisión")

# ── SEGMENTACIÓN ────────────────────────────────────────────
elif seccion == "📊 Segmentación":
    st.title("📊 Segmentación de Clientes")
    st.info("🚧 En construcción...")

# ── PREDICCIÓN ──────────────────────────────────────────────
elif seccion == "🔮 Predicción":
    st.title("🔮 Predicción de Segmento")
    st.info("🚧 En construcción...")

# ── COMPARACIÓN ─────────────────────────────────────────────
elif seccion == "⚖️ Comparación":
    st.title("⚖️ Comparación Naturales vs Jurídicos")
    st.info("🚧 En construcción...")
