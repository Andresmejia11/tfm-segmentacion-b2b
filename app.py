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
    k4.metric("Modelos evaluados", "2")

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
    st.markdown("---")

    # Selector tipo de cliente
    tipo = st.radio("Selecciona el tipo de cliente:", 
                    ["🔵 Naturales", "🟣 Jurídicos"], 
                    horizontal=True)
    
    tipo_key = "NATURAL" if "Naturales" in tipo else "JURIDICO"

    # Selector de análisis
    analisis = st.selectbox("¿Qué análisis quieres ver?", [
        "👥 Perfiles de clusters",
        "📈 Plano FM (Frecuencia vs Monto)",
        "📉 Método del codo",
        "🔵 PCA",
        "🔍 DBSCAN",
        "🧠 SOM"
    ])

    st.markdown("---")

    if analisis == "👥 Perfiles de clusters":
        st.subheader(f"Perfiles de clusters · {tipo_key.title()}")
        st.info("🚧 En construcción...")

    elif analisis == "📈 Plano FM (Frecuencia vs Monto)":
        st.subheader("Plano FM · Frecuencia vs Monto")

        ventas_agg2 = ventas.groupby("ID").agg(
            TOTAL_VENTAS=("IMPORTE", "sum"),
            PROMEDIO_VENTA=("IMPORTE", "mean"),
        ).reset_index()
        consultas_agg2 = consultas.groupby("ID").agg(
            NUM_CONSULTAS=("IDCONSUMO", "count")
        ).reset_index()
        df2 = clientes.merge(ventas_agg2, on="ID", how="left").merge(consultas_agg2, on="ID", how="left")
        df2["NUM_CONSULTAS"] = df2["NUM_CONSULTAS"].fillna(0).astype(int)
        naturales2 = ["PERSONA FISICA", "EMPRESARIO"]
        df2["TIPO_CLIENTE"] = df2["FORMAJURIDICA"].apply(
            lambda x: "NATURAL" if x in naturales2 else "JURIDICO"
        )
        df_nat2 = df2[df2["TIPO_CLIENTE"] == "NATURAL"]
        df_jur2 = df2[df2["TIPO_CLIENTE"] == "JURIDICO"]

        # Filtro outliers p95
        df_nat_c = df_nat2[
            (df_nat2["NUM_COMPRAS"] <= df_nat2["NUM_COMPRAS"].quantile(0.95)) &
            (df_nat2["TOTAL_VENTAS"] <= df_nat2["TOTAL_VENTAS"].quantile(0.95))
        ]
        df_jur_c = df_jur2[
            (df_jur2["NUM_COMPRAS"] <= df_jur2["NUM_COMPRAS"].quantile(0.95)) &
            (df_jur2["TOTAL_VENTAS"] <= df_jur2["TOTAL_VENTAS"].quantile(0.95))
        ]

        escala = st.radio("Escala del eje Y:", ["Normal", "Logarítmica"], horizontal=True)

        if tipo_key == "NATURAL":
            df_plot = df_nat_c
            color = "mediumslateblue"
        else:
            df_plot = df_jur_c
            color = "skyblue"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["NUM_COMPRAS"],
            y=df_plot["TOTAL_VENTAS"],
            mode="markers",
            marker=dict(color=color, opacity=0.5, size=6),
            name=tipo_key
        ))
        fig.update_layout(
            title=f"Plano FM · {tipo_key.title()}",
            xaxis_title="Frecuencia (Nº Compras)",
            yaxis_title="Monto Total de Ventas",
            yaxis_type="log" if escala == "Logarítmica" else "linear",
            template="simple_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla estadística
        st.markdown("**Estadísticas descriptivas**")
        st.dataframe(
            df_plot[["NUM_COMPRAS", "TOTAL_VENTAS"]].describe().round(2),
            use_container_width=True
        )

    elif analisis == "📉 Método del codo":
        st.subheader("Método del codo")
        st.info("🚧 En construcción...")

    elif analisis == "🔵 PCA":
        st.subheader(f"PCA · {tipo_key.title()}")
        st.info("🚧 En construcción...")

    elif analisis == "🔍 DBSCAN":
        st.subheader(f"DBSCAN · {tipo_key.title()}")
        st.info("🚧 En construcción...")

    elif analisis == "🧠 SOM":
        st.subheader(f"SOM · {tipo_key.title()}")
        st.info("🚧 En construcción...")

# ── PREDICCIÓN ──────────────────────────────────────────────
elif seccion == "🔮 Predicción":
    st.title("🔮 Predicción de Segmento")
    st.info("🚧 En construcción...")

# ── COMPARACIÓN ─────────────────────────────────────────────
elif seccion == "⚖️ Comparación":
    st.title("⚖️ Comparación Naturales vs Jurídicos")
    st.info("🚧 En construcción...")
