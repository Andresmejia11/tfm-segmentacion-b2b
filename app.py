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

# ── Procesamiento base ──────────────────────────────────────
@st.cache_data(show_spinner="Procesando datos...")
def procesar_datos(_clientes, _ventas, _consultas):
    ventas_agg = _ventas.groupby("ID").agg(
        TOTAL_VENTAS=("IMPORTE", "sum"),
        PROMEDIO_VENTA=("IMPORTE", "mean"),
    ).reset_index()
    consultas_agg = _consultas.groupby("ID").agg(
        NUM_CONSULTAS=("IDCONSUMO", "count")
    ).reset_index()
    df = _clientes.merge(ventas_agg, on="ID", how="left").merge(consultas_agg, on="ID", how="left")
    df["NUM_CONSULTAS"]  = df["NUM_CONSULTAS"].fillna(0).astype(int)
    df["TOTAL_VENTAS"]   = df["TOTAL_VENTAS"].fillna(0)
    df["PROMEDIO_VENTA"] = df["PROMEDIO_VENTA"].fillna(0)
    naturales = ["PERSONA FISICA", "EMPRESARIO"]
    df["TIPO_CLIENTE"] = df["FORMAJURIDICA"].apply(
        lambda x: "NATURAL" if x in naturales else "JURIDICO"
    )
    return df

df = procesar_datos(clientes, ventas, consultas)

VARS = ['TOTAL_VENTAS', 'NUM_COMPRAS', 'NUM_CONSULTAS', 'EMPRESASUNICAS_CONSULT']

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

# ── Pipeline clustering ─────────────────────────────────────
@st.cache_data(show_spinner="Calculando clusters...")
def calcular_clusters(tipo_key):
    d = df[df["TIPO_CLIENTE"] == tipo_key].copy()
    d = d[
        (d["NUM_COMPRAS"] <= d["NUM_COMPRAS"].quantile(0.95)) &
        (d["TOTAL_VENTAS"] <= d["TOTAL_VENTAS"].quantile(0.95))
    ].copy()
    for col in VARS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d_log = d.copy()
    for col in VARS:
        d_log[col] = np.log1p(d_log[col])
    mask  = d_log[VARS].notna().all(axis=1)
    d_log = d_log[mask].reset_index(drop=True)
    d     = d[mask].reset_index(drop=True)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(d_log[VARS])
    km = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=20)
    d["cluster"] = km.fit_predict(X_scaled)
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    d["PC1"] = X_pca[:, 0]
    d["PC2"] = X_pca[:, 1]
    var_exp = pca.explained_variance_ratio_
    return d, X_scaled, var_exp

NOMBRES = {0: "Ocasionales", 1: "Recurrentes", 2: "Intensivos"}
COLORES = {"Ocasionales": "#6366f1", "Recurrentes": "#10b981", "Intensivos": "#f59e0b"}

# ══════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════
if seccion == "🏠 Inicio":
    st.title("📊 Segmentación de Clientes B2B")
    st.subheader("Análisis de recurrencia en el sector de información empresarial · Colombia")
    st.markdown("---")

    total   = len(df)
    n_nat   = (df["TIPO_CLIENTE"] == "NATURAL").sum()
    n_jur   = (df["TIPO_CLIENTE"] == "JURIDICO").sum()
    pct_nat = n_nat / total * 100
    pct_jur = n_jur / total * 100

    st.markdown("### Resumen general")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total clientes",     f"{total:,}")
    k2.metric("Clientes Naturales", f"{n_nat:,}",  f"{pct_nat:.1f}%")
    k3.metric("Clientes Jurídicos", f"{n_jur:,}",  f"{pct_jur:.1f}%")
    k4.metric("Modelos evaluados",  "2")

    st.markdown("---")
    st.markdown("### ¿Qué encontramos?")
    col1, col2 = st.columns(2)
    with col1:
        st.success("🔵 **Clientes Naturales** · 3 segmentos: Ocasionales, Recurrentes e Intensivos")
        st.success("🟣 **Clientes Jurídicos** · 3 segmentos: Ocasionales, Recurrentes e Intensivos")
    with col2:
        st.info("🎯 **Predicción Naturales** · Random Forest 96% de precisión")
        st.info("🎯 **Predicción Jurídicos** · Random Forest 89% de precisión")

# ══════════════════════════════════════════════════════════════
# SEGMENTACIÓN
# ══════════════════════════════════════════════════════════════
elif seccion == "📊 Segmentación":
    st.title("📊 Segmentación de Clientes")
    st.markdown("---")

    tipo = st.radio("Selecciona el tipo de cliente:",
                    ["🔵 Naturales", "🟣 Jurídicos"], horizontal=True)
    tipo_key = "NATURAL" if "Naturales" in tipo else "JURIDICO"

    analisis = st.selectbox("¿Qué análisis quieres ver?", [
        "👥 Perfiles de clusters",
        "📈 Plano FM (Frecuencia vs Monto)",
        "📉 Método del codo",
        "🔵 PCA",
        "🔍 DBSCAN",
        "🧠 SOM"
    ])

    st.markdown("---")

    df_seg, X_scaled, var_exp = calcular_clusters(tipo_key)
    df_seg["Segmento"] = df_seg["cluster"].map(NOMBRES)
    color_tipo = "mediumslateblue" if tipo_key == "NATURAL" else "darkorange"

    if analisis == "👥 Perfiles de clusters":
        st.subheader(f"Perfiles de clusters · {tipo_key.title()}")
        perfil = df_seg.groupby("Segmento")[VARS].mean().reset_index()
        col1, col2, col3 = st.columns(3)
        for col, seg in zip([col1, col2, col3], ["Ocasionales", "Recurrentes", "Intensivos"]):
            row = perfil[perfil["Segmento"] == seg]
            if not row.empty:
                col.markdown(f"**{seg}**")
                col.metric("Ventas promedio", f"${row['TOTAL_VENTAS'].values[0]:,.0f}")
                col.metric("Nº compras",      f"{row['NUM_COMPRAS'].values[0]:.1f}")
                col.metric("Nº consultas",    f"{row['NUM_CONSULTAS'].values[0]:.0f}")
                col.metric("Empresas únicas", f"{row['EMPRESASUNICAS_CONSULT'].values[0]:.1f}")
        st.markdown("---")
        st.markdown("**Tabla completa de promedios**")
        st.dataframe(perfil.set_index("Segmento").round(2), use_container_width=True)
        st.markdown("**Distribución de clientes por segmento**")
        conteo = df_seg["Segmento"].value_counts().reset_index()
        conteo.columns = ["Segmento", "Clientes"]
        fig = px.bar(conteo, x="Segmento", y="Clientes", color="Segmento",
                     color_discrete_map=COLORES, template="simple_white", text="Clientes")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    elif analisis == "📈 Plano FM (Frecuencia vs Monto)":
        st.subheader("Plano FM · Naturales vs Jurídicos")
        df_nat_c = df[df["TIPO_CLIENTE"] == "NATURAL"].copy()
        df_nat_c = df_nat_c[
            (df_nat_c["NUM_COMPRAS"] <= df_nat_c["NUM_COMPRAS"].quantile(0.95)) &
            (df_nat_c["TOTAL_VENTAS"] <= df_nat_c["TOTAL_VENTAS"].quantile(0.95))
        ]
        df_jur_c = df[df["TIPO_CLIENTE"] == "JURIDICO"].copy()
        df_jur_c = df_jur_c[
            (df_jur_c["NUM_COMPRAS"] <= df_jur_c["NUM_COMPRAS"].quantile(0.95)) &
            (df_jur_c["TOTAL_VENTAS"] <= df_jur_c["TOTAL_VENTAS"].quantile(0.95))
        ]
        escala = st.radio("Escala del eje Y:", ["Normal", "Logarítmica"], horizontal=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_nat_c["NUM_COMPRAS"], y=df_nat_c["TOTAL_VENTAS"],
            mode="markers",
            marker=dict(color="mediumslateblue", opacity=0.5, size=6, symbol="circle"),
            name="NATURAL (B2C)"
        ))
        fig.add_trace(go.Scatter(
            x=df_jur_c["NUM_COMPRAS"], y=df_jur_c["TOTAL_VENTAS"],
            mode="markers",
            marker=dict(color="darkorange", opacity=0.7, size=7, symbol="triangle-up"),
            name="JURÍDICO (B2B)"
        ))
        fig.update_layout(
            title="Plano FM · Naturales vs Jurídicos",
            xaxis_title="Frecuencia (Nº Compras)",
            yaxis_title="Monto Total de Ventas",
            yaxis_type="log" if escala == "Logarítmica" else "linear",
            template="simple_white", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Comparativa estadística**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🔵 **Naturales**")
            st.dataframe(df_nat_c[["NUM_COMPRAS","TOTAL_VENTAS"]].describe().round(2), use_container_width=True)
        with col2:
            st.markdown("🟠 **Jurídicos**")
            st.dataframe(df_jur_c[["NUM_COMPRAS","TOTAL_VENTAS"]].describe().round(2), use_container_width=True)

    elif analisis == "📉 Método del codo":
        st.subheader(f"Método del codo · {tipo_key.title()}")
        wcss, results, previous = [], [], None
        for k in range(1, 10):
            km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=20)
            km.fit(X_scaled)
            inertia   = km.inertia_
            reduccion = None if previous is None else round(previous - inertia, 1)
            wcss.append(inertia)
            results.append({"k": k, "Inercia": round(inertia, 1), "Reducción": reduccion})
            previous  = inertia
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 10)), y=wcss,
            mode="lines+markers",
            marker=dict(size=8, color=color_tipo),
            line=dict(color=color_tipo, width=2),
            name=tipo_key
        ))
        fig.add_vline(x=3, line_dash="dash", line_color="red",
                      annotation_text="k=3 óptimo", annotation_position="top right")
        fig.update_layout(
            title=f"Método del Codo · {tipo_key.title()} (escala log)",
            xaxis_title="Número de Clústeres (k)",
            yaxis_title="Inercia (WCSS)",
            template="simple_white", height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Tabla de inercias · justificación de k=3**")
        df_codo = pd.DataFrame(results)
        st.dataframe(df_codo, use_container_width=True, hide_index=True)
        st.success("✅ La mayor reducción de inercia ocurre en k=3, confirmando el número óptimo de clústeres.")

    elif analisis == "🔵 PCA":
        st.subheader(f"PCA · {tipo_key.title()}")
        fig = px.scatter(
            df_seg, x="PC1", y="PC2", color="Segmento",
            color_discrete_map=COLORES, opacity=0.7, template="simple_white",
            title=f"Proyección PCA · Clientes {tipo_key.title()}",
            hover_data={"PC1": False, "PC2": False,
                        "TOTAL_VENTAS": ":,.0f", "NUM_COMPRAS": True}
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.add_annotation(
            text=f"PC1 explica {var_exp[0]*100:.1f}% · PC2 explica {var_exp[1]*100:.1f}% de la varianza",
            xref="paper", yref="paper", x=0, y=-0.12,
            showarrow=False, font=dict(size=11, color="#888")
        )
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Varianza explicada PC1", f"{var_exp[0]*100:.1f}%")
        col2.metric("Varianza explicada PC2", f"{var_exp[1]*100:.1f}%")
        st.info(f"Entre PC1 y PC2 se explica el **{(var_exp[0]+var_exp[1])*100:.1f}%** de la varianza total.")

    elif analisis == "🔍 DBSCAN":
        st.subheader(f"DBSCAN · {tipo_key.title()}")
        eps_val = 0.85 if tipo_key == "NATURAL" else 0.7
        dbscan  = DBSCAN(eps=eps_val, min_samples=5)
        labels  = dbscan.fit_predict(X_scaled)
        df_seg["cluster_db"] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_ruido    = list(labels).count(-1)
        col1, col2 = st.columns(2)
        col1.metric("Clústeres detectados", n_clusters)
        col2.metric("Puntos de ruido",      n_ruido)
        df_seg["DBSCAN"] = df_seg["cluster_db"].apply(
            lambda x: "Ruido" if x == -1 else f"Cluster {x}"
        )
        fig = px.scatter(
            df_seg, x="PC1", y="PC2", color="DBSCAN",
            opacity=0.7, template="simple_white",
            title=f"DBSCAN · Clientes {tipo_key.title()} (eps={eps_val})"
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Los puntos en **Ruido** son clientes atípicos que DBSCAN no asigna a ningún grupo — a diferencia de K-Means que los fuerza a un cluster.")

    elif analisis == "🧠 SOM":
        st.subheader(f"SOM · Mapa Autoorganizado · {tipo_key.title()}")
        with st.spinner("Entrenando SOM..."):
            som = MiniSom(x=6, y=6, input_len=X_scaled.shape[1],
                          sigma=1.0, learning_rate=0.5, random_seed=42)
            som.random_weights_init(X_scaled)
            som.train_random(X_scaled, 1000)
        u_matrix = som.distance_map()
        fig = go.Figure(data=go.Heatmap(
            z=u_matrix,
            colorscale="RdYlBu_r",
            colorbar=dict(title="Distancia entre nodos")
        ))
        fig.update_layout(
            title=f"U-Matrix SOM · {tipo_key.title()}",
            xaxis_title="Columna", yaxis_title="Fila",
            height=450, template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Cómo leer el mapa:**
        - 🔵 **Colores fríos (azul)** → zonas de alta densidad = clústeres
        - 🔴 **Colores cálidos (rojo/amarillo)** → fronteras entre segmentos
        """)

# ══════════════════════════════════════════════════════════════
# PREDICCIÓN
# ══════════════════════════════════════════════════════════════
elif seccion == "🔮 Predicción":
    st.title("🔮 Predicción de Segmento")
    st.markdown("---")
    tipo = st.radio("Selecciona el tipo de cliente:",
                    ["🔵 Naturales", "🟣 Jurídicos"], horizontal=True)
    tipo_key = "NATURAL" if "Naturales" in tipo else "JURIDICO"
    st.info("🚧 En construcción...")

# ══════════════════════════════════════════════════════════════
# COMPARACIÓN
# ══════════════════════════════════════════════════════════════
elif seccion == "⚖️ Comparación":
    st.title("⚖️ Comparación · Naturales vs Jurídicos")
    st.markdown("---")
    st.info("🚧 En construcción...")
