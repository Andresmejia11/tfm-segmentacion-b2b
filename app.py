import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from minisom import MiniSom
import warnings
warnings.filterwarnings("ignore")

# ── Carga de datos ─────────────────────────────────────────
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

# ── Procesamiento exacto igual al notebook ─────────────────
@st.cache_data(show_spinner="Procesando datos...")
def procesar_datos(_clientes, _ventas, _consultas):
    ventas_agg = _ventas.groupby("ID").agg(
        TOTAL_VENTAS=("IMPORTE", "sum"),
        PROMEDIO_VENTA=("IMPORTE", "mean"),
        NUM_VENTAS=("IMPORTE", "count")
    ).reset_index()
    consultas_agg = _consultas.groupby("ID").agg(
        NUM_CONSULTAS=("IDCONSUMO", "count")
    ).reset_index()
    df = _clientes.merge(ventas_agg, on="ID", how="left").merge(consultas_agg, on="ID", how="left")
    df["NUM_CONSULTAS"]  = df["NUM_CONSULTAS"].fillna(0).astype(int)
    df["TOTAL_VENTAS"]   = df["TOTAL_VENTAS"].fillna(0)
    df["PROMEDIO_VENTA"] = df["PROMEDIO_VENTA"].fillna(0)
    df["NUM_VENTAS"]     = df["NUM_VENTAS"].fillna(0)
    df["FECHA_REGISTRO"] = pd.to_datetime(df["FECHA_REGISTRO"], dayfirst=True, errors="coerce")
    df["FECHA_CLIENTE"]  = pd.to_datetime(df["FECHA_CLIENTE"],  dayfirst=True, errors="coerce")
    naturales = ["PERSONA FISICA", "EMPRESARIO"]
    df["TIPO_CLIENTE"] = df["FORMAJURIDICA"].apply(
        lambda x: "NATURAL" if x in naturales else "JURIDICO"
    )
    df = df.drop(columns=["IMPORTE_COMPRAS", "NUM_VENTAS", "CONSUMOSTOTAL"], errors="ignore")
    return df

df = procesar_datos(clientes, ventas, consultas)

VARS = ['TOTAL_VENTAS', 'NUM_COMPRAS', 'NUM_CONSULTAS', 'EMPRESASUNICAS_CONSULT']

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
    d_log = d_log.dropna(subset=VARS)
    d     = d.loc[d_log.index].reset_index(drop=True)
    d_log = d_log.reset_index(drop=True)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(d_log[VARS])
    km = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=20)
    d["cluster"] = km.fit_predict(X_scaled)
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    d["PC1"] = X_pca[:, 0]
    d["PC2"] = X_pca[:, 1]
    return d, X_scaled, pca.explained_variance_ratio_

NOMBRES = {0: "Ocasionales", 1: "Recurrentes", 2: "Intensivos"}
COLORES = {"Ocasionales": "#6366f1", "Recurrentes": "#10b981", "Intensivos": "#f59e0b"}

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
    tipo_key   = "NATURAL" if "Naturales" in tipo else "JURIDICO"
    color_tipo = "mediumslateblue" if tipo_key == "NATURAL" else "darkorange"

    analisis = st.selectbox("¿Qué análisis quieres ver?", [
        "👥 Perfiles de clusters",
        "📈 Plano FM (Frecuencia vs Monto)",
        "🔵 PCA",
        "🔍 DBSCAN",
        "🧠 SOM"
    ])
    st.markdown("---")

    df_seg, X_scaled, var_exp = calcular_clusters(tipo_key)
    df_seg["Segmento"] = df_seg["cluster"].map(NOMBRES)

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
        with st.expander("¿Cómo interpretar estos perfiles?"):
            st.markdown("""
            Cada segmento agrupa clientes con comportamiento similar:
            - **Ocasionales:** compran poco y generan bajos ingresos. Son clientes de bajo compromiso.
            - **Recurrentes:** compran con frecuencia media y volumen de consultas moderado. Son el segmento más estable.
            - **Intensivos:** alta frecuencia de compra, alto monto y muchas consultas. Son los clientes más valiosos.
            """)

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
        with st.expander("¿Cómo interpretar el Plano FM?"):
            st.markdown("""
            El Plano FM (Frecuencia vs Monto) permite visualizar el comportamiento de compra:
            - **Eje X (Frecuencia):** cuántas veces ha comprado el cliente.
            - **Eje Y (Monto):** cuánto dinero ha generado en total.
            - Los clientes en la esquina superior derecha son los más valiosos — compran mucho y frecuentemente.
            - La escala logarítmica ayuda a visualizar mejor cuando hay grandes diferencias entre clientes.
            """)

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
        with st.expander("¿Cómo interpretar el PCA?"):
            st.markdown("""
            El Análisis de Componentes Principales (PCA) reduce las 4 variables a 2 dimensiones para poder visualizarlas:
            - Cada punto es un cliente.
            - El color indica a qué segmento pertenece.
            - Los grupos bien separados confirman que los clusters son distintos entre sí.
            - Cuanto mayor sea la varianza explicada, más fiel es la representación.
            """)

    elif analisis == "🔍 DBSCAN":
        st.subheader(f"DBSCAN · {tipo_key.title()}")
        eps_val = 0.85 if tipo_key == "NATURAL" else 0.7
        labels  = DBSCAN(eps=eps_val, min_samples=5).fit_predict(X_scaled)
        df_seg["DBSCAN"] = ["Ruido" if x == -1 else f"Cluster {x}" for x in labels]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_ruido    = list(labels).count(-1)
        col1, col2 = st.columns(2)
        col1.metric("Clústeres detectados", n_clusters)
        col2.metric("Puntos de ruido",      n_ruido)
        fig = px.scatter(df_seg, x="PC1", y="PC2", color="DBSCAN",
                         opacity=0.7, template="simple_white",
                         title=f"DBSCAN · {tipo_key.title()} (eps={eps_val})")
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Los puntos en **Ruido** son clientes atípicos que DBSCAN no asigna a ningún grupo — a diferencia de K-Means que los fuerza a un cluster.")
        with st.expander("¿Cómo interpretar el DBSCAN?"):
            st.markdown("""
            DBSCAN es un algoritmo que detecta clusters basándose en densidad, sin necesidad de definir k de antemano:
            - Agrupa puntos que están cerca entre sí.
            - Los puntos **Ruido** son clientes con comportamiento tan atípico que no encajan en ningún grupo.
            - Se usa como validación de los clusters encontrados por K-Means.
            - Si DBSCAN encuentra los mismos grupos, confirma que la segmentación es sólida.
            """)

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
            zsmooth="best",
            colorbar=dict(title="Distancia entre nodos"),
            showscale=True
        ))
        fig.update_layout(
            title=f"SOM - U-Matrix · {tipo_key.title()}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500,
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("¿Cómo interpretar la U-Matrix del SOM?"):
            st.markdown("""
            El Mapa Autoorganizado (SOM) es una red neuronal no supervisada que organiza los clientes en una cuadrícula:
            - 🔵 **Colores fríos (azul)** → nodos similares entre sí = zonas densas = clústeres.
            - 🔴 **Colores cálidos (rojo/amarillo)** → mayor distancia entre nodos = fronteras naturales entre segmentos.
            - Las zonas azules separadas por zonas cálidas confirman la existencia de grupos diferenciados.
            - Es una validación visual independiente de K-Means y DBSCAN.
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
    st.markdown("---")

    # ── Umbrales exactos del notebook ──────────────────────────
    def segmentar_nat(x):
        if x <= 15:    return "MUY_BAJO"
        elif x <= 35:  return "BAJO"
        elif x <= 65:  return "MEDIO"
        elif x <= 105: return "ALTO"
        else:          return "VIP"

    def segmentar_jur(x):
        if x <= 35:    return "MUY_BAJO"
        elif x <= 70:  return "BAJO"
        elif x <= 214: return "MEDIO"
        elif x <= 500: return "ALTO"
        else:          return "VIP"

    # ── Cargar modelos desde GitHub (.pkl) ─────────────────────
    @st.cache_resource(show_spinner="Cargando modelos...")
    def cargar_modelos():
        import joblib, requests, io
        base = "https://raw.githubusercontent.com/Andresmejia11/tfm-segmentacion-b2b/main/"
        modelos = {}
        for nombre in ["rf_naturales", "lr_naturales", "rf_juridicos", "lr_juridicos"]:
            r = requests.get(base + nombre + ".pkl")
            modelos[nombre] = joblib.load(io.BytesIO(r.content))
        return modelos

    modelos = cargar_modelos()
    rf_model = modelos["rf_naturales"] if tipo_key == "NATURAL" else modelos["rf_juridicos"]
    lr_model = modelos["lr_naturales"] if tipo_key == "NATURAL" else modelos["lr_juridicos"]

    # ── Calcular métricas con el modelo cargado ─────────────────
    @st.cache_data(show_spinner="Calculando métricas...")
    def calcular_metricas(tipo_key):
        import warnings
        warnings.filterwarnings("ignore")

        d = df[df["TIPO_CLIENTE"] == tipo_key].copy()

        if tipo_key == "NATURAL":
            d["segmento_final"] = d["TOTAL_VENTAS"].apply(segmentar_nat)
            y = d["segmento_final"]
            numericas   = ["NUM_COMPRAS", "NUM_CONSULTAS", "EMPRESASUNICAS_CONSULT",
                           "DIASCLIENTE", "PROMEDIO_VENTA", "CLIENTEPORCAMPAÑAEMAIL"]
            categoricas = ["CANAL_REGISTRO", "DEPARTAMENTO", "ANTIGUEDAD",
                           "DESC_SECTOR", "ESTADO", "TAMAÑO", "FORMAJURIDICA", "SECTOR"]
        else:
            d["segmento_finaljur"] = d["TOTAL_VENTAS"].apply(segmentar_jur)
            y = d["segmento_finaljur"]
            numericas   = ["NUM_COMPRAS", "NUM_CONSULTAS", "EMPRESASUNICAS_CONSULT",
                           "DIASCLIENTE", "PROMEDIO_VENTA", "CLIENTEPORCAMPAÑAEMAIL"]
            categoricas = ["CANAL_REGISTRO", "DEPARTAMENTO", "ANTIGUEDAD",
                           "DESC_SECTOR", "ESTADO", "TAMAÑO"]

        numericas   = [c for c in numericas   if c in d.columns]
        categoricas = [c for c in categoricas if c in d.columns]
        X = d[numericas + categoricas].copy()
        X_encoded = pd.get_dummies(X, drop_first=True).astype(int)
        if tipo_key == "JURIDICO":
            cols_drop = [c for c in X_encoded.columns if "EMPRESASUNICAS_CONSULT" in c]
            X_encoded = X_encoded.drop(columns=cols_drop, errors="ignore")

        _, X_test, _, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        rf  = modelos["rf_naturales" if tipo_key == "NATURAL" else "rf_juridicos"]
        lr  = modelos["lr_naturales" if tipo_key == "NATURAL" else "lr_juridicos"]

        # Alinear columnas con las que el modelo conoce
        X_test_rf = X_test.reindex(columns=rf.feature_names_in_, fill_value=0)
        from sklearn.preprocessing import StandardScaler as SS
        scaler2    = SS()
        X_test_sc  = scaler2.fit_transform(X_test_rf)

        y_pred_rf = rf.predict(X_test_rf)
        y_pred_lr = lr.predict(X_test_sc)

        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

        importancias = pd.Series(
            rf.feature_importances_, index=rf.feature_names_in_
        ).sort_values(ascending=False).head(10)

        return report_rf, report_lr, importancias, X_encoded.columns.tolist()

    report_rf, report_lr, importancias, feature_cols = calcular_metricas(tipo_key)

    # ── Resultados exactos del notebook (hardcoded) ────────────
    RESULTADOS = {
        "NATURAL": {
            "acc_rf": 0.96, "acc_lr": 0.90,
            "rf": [
                {"Segmento": "MUY_BAJO", "Precisión": "100%", "Recall": "100%", "F1-Score": "100%", "Soporte": 212},
                {"Segmento": "BAJO",     "Precisión": "96%",  "Recall": "96%",  "F1-Score": "96%",  "Soporte": 158},
                {"Segmento": "MEDIO",    "Precisión": "95%",  "Recall": "94%",  "F1-Score": "95%",  "Soporte": 143},
                {"Segmento": "ALTO",     "Precisión": "93%",  "Recall": "94%",  "F1-Score": "93%",  "Soporte": 99},
                {"Segmento": "VIP",      "Precisión": "97%",  "Recall": "96%",  "F1-Score": "96%",  "Soporte": 75},
            ],
            "lr": [
                {"Segmento": "MUY_BAJO", "Precisión": "97%",  "Recall": "99%",  "F1-Score": "98%",  "Soporte": 212},
                {"Segmento": "BAJO",     "Precisión": "89%",  "Recall": "90%",  "F1-Score": "90%",  "Soporte": 158},
                {"Segmento": "MEDIO",    "Precisión": "88%",  "Recall": "84%",  "F1-Score": "86%",  "Soporte": 143},
                {"Segmento": "ALTO",     "Precisión": "82%",  "Recall": "82%",  "F1-Score": "82%",  "Soporte": 99},
                {"Segmento": "VIP",      "Precisión": "91%",  "Recall": "89%",  "F1-Score": "90%",  "Soporte": 75},
            ],
            "importancias": {
                "PROMEDIO_VENTA": 0.61, "NUM_COMPRAS": 0.12, "NUM_CONSULTAS": 0.07,
                "EMPRESASUNICAS_CONSULT": 0.06, "DIASCLIENTE": 0.04,
                "TOTAL_VENTAS": 0.03, "ANTIGUEDAD": 0.02,
                "CANAL_REGISTRO_WEB": 0.01, "DEPARTAMENTO_BOGOTA": 0.01, "TAMAÑO_PEQUEÑA": 0.01
            }
        },
        "JURIDICO": {
            "acc_rf": 0.89, "acc_lr": 0.79,
            "rf": [
                {"Segmento": "MUY_BAJO", "Precisión": "80%",  "Recall": "100%", "F1-Score": "89%",  "Soporte": 144},
                {"Segmento": "BAJO",     "Precisión": "75%",  "Recall": "61%",  "F1-Score": "67%",  "Soporte": 112},
                {"Segmento": "MEDIO",    "Precisión": "80%",  "Recall": "76%",  "F1-Score": "78%",  "Soporte": 108},
                {"Segmento": "ALTO",     "Precisión": "86%",  "Recall": "86%",  "F1-Score": "86%",  "Soporte": 80},
                {"Segmento": "VIP",      "Precisión": "100%", "Recall": "91%",  "F1-Score": "95%",  "Soporte": 85},
            ],
            "lr": [
                {"Segmento": "MUY_BAJO", "Precisión": "100%", "Recall": "99%",  "F1-Score": "100%", "Soporte": 144},
                {"Segmento": "BAJO",     "Precisión": "94%",  "Recall": "94%",  "F1-Score": "94%",  "Soporte": 112},
                {"Segmento": "MEDIO",    "Precisión": "89%",  "Recall": "82%",  "F1-Score": "85%",  "Soporte": 108},
                {"Segmento": "ALTO",     "Precisión": "67%",  "Recall": "84%",  "F1-Score": "74%",  "Soporte": 80},
                {"Segmento": "VIP",      "Precisión": "91%",  "Recall": "78%",  "F1-Score": "84%",  "Soporte": 85},
            ],
            "importancias": {
                "PROMEDIO_VENTA": 0.334, "NUM_COMPRAS": 0.164, "NUM_CONSULTAS": 0.136,
                "EMPRESASUNICAS_CONSULT": 0.124, "DIASCLIENTE": 0.036,
                "DEPARTAMENTO_BOGOTA": 0.013, "CANAL_REGISTRO_WEB": 0.013,
                "TAMAÑO_PEQUEÑA": 0.013, "DESC_SECTOR_COMERCIO": 0.010, "TAMAÑO_MEDIANA": 0.010
            }
        }
    }

    # ── Métricas comparativas ───────────────────────────────────
    st.markdown("### Comparación de modelos")
    acc_rf = report_rf["accuracy"]
    acc_lr = report_lr["accuracy"]

    col1, col2 = st.columns(2)
    col1.metric("🌲 Random Forest · Precisión global", f"{acc_rf:.0%}",
                delta="Modelo recomendado" if acc_rf > acc_lr else None)
    col2.metric("📈 Regresión Logística · Precisión global", f"{acc_lr:.0%}")

    with st.expander("¿Por qué Random Forest es mejor en este caso?"):
        st.markdown("""
        - **Random Forest** captura relaciones no lineales entre variables, lo que lo hace más adecuado para segmentación de clientes.
        - **Regresión Logística** asume relaciones lineales, lo que limita su capacidad para distinguir segmentos complejos como ALTO y VIP.
        - La diferencia es especialmente notable en los segmentos de mayor valor, donde RF logra identificarlos correctamente con mayor frecuencia.
        """)

    st.markdown("---")

    # ── Tabla de métricas por segmento ─────────────────────────
    modelo_sel = st.selectbox("Ver detalle de métricas por segmento:",
                               ["🌲 Random Forest", "📈 Regresión Logística"])
    report_sel = report_rf if "Random" in modelo_sel else report_lr

    segmentos = ["MUY_BAJO", "BAJO", "MEDIO", "ALTO", "VIP"]
    filas = []
    for seg in segmentos:
        if seg in report_sel:
            r = report_sel[seg]
            filas.append({
                "Segmento":  seg,
                "Precisión": f"{r['precision']:.0%}",
                "Recall":    f"{r['recall']:.0%}",
                "F1-Score":  f"{r['f1-score']:.0%}",
                "Soporte":   int(r['support'])
            })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Importancia de variables ────────────────────────────────
    st.markdown("### Variables más importantes · Random Forest")
    fig_imp = go.Figure(go.Bar(
        x=importancias.values,
        y=importancias.index,
        orientation="h",
        marker=dict(color="mediumslateblue" if tipo_key == "NATURAL" else "darkorange")
    ))
    fig_imp.update_layout(
        title=f"Top 10 variables · {tipo_key.title()}",
        xaxis_title="Importancia",
        yaxis=dict(autorange="reversed"),
        template="simple_white",
        height=400
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("¿Cómo usar estas variables para tomar decisiones?"):
        st.markdown("""
        Las variables más importantes indican qué factores definen el valor de un cliente:
        - **PROMEDIO_VENTA alto** → cliente de alto valor. Priorizar fidelización y atención personalizada.
        - **NUM_COMPRAS alto** → cliente recurrente. Ideal para programas de lealtad.
        - **NUM_CONSULTAS alto** → cliente con interés activo. Oportunidad para ampliar oferta de productos.
        - **EMPRESASUNICAS_CONSULT alto** → cliente que consulta muchas empresas. Potencial de expansión de cartera.

        **Estrategia sugerida por segmento:**
        - 🔴 **MUY_BAJO / BAJO:** campañas de activación y reenganche.
        - 🟡 **MEDIO:** incentivos para aumentar frecuencia de compra.
        - 🟢 **ALTO / VIP:** atención prioritaria, descuentos exclusivos, gestor dedicado.
        """)

    st.markdown("---")

    # ── Predictor individual ────────────────────────────────────
    st.markdown("### 🔍 Predice el segmento de un cliente nuevo")
    st.markdown("Ingresa los datos del cliente y el modelo Random Forest predecirá su segmento.")

    c1, c2 = st.columns(2)
    with c1:
        total_ventas   = st.number_input("Ventas totales (COP)",      min_value=0.0, value=50.0,  step=10.0)
        promedio_venta = st.number_input("Promedio por venta (COP)",  min_value=0.0, value=25.0,  step=5.0)
        num_compras    = st.number_input("Número de compras",         min_value=0,   value=2,     step=1)
    with c2:
        num_consultas  = st.number_input("Número de consultas",       min_value=0,   value=5,     step=1)
        emp_unicas     = st.number_input("Empresas únicas consultadas",min_value=0,   value=3,     step=1)
        antiguedad     = st.number_input("Antigüedad (días)",         min_value=0,   value=365,   step=30)

    COLORES_SEG = {
        "MUY_BAJO": "#94a3b8",
        "BAJO":     "#60a5fa",
        "MEDIO":    "#34d399",
        "ALTO":     "#f59e0b",
        "VIP":      "#ef4444"
    }

    if st.button("🔮 Predecir segmento"):
        X_new = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)
        for col, val in [("TOTAL_VENTAS", total_ventas), ("PROMEDIO_VENTA", promedio_venta),
                         ("NUM_COMPRAS", num_compras),   ("NUM_CONSULTAS", num_consultas),
                         ("EMPRESASUNICAS_CONSULT", emp_unicas), ("ANTIGUEDAD", antiguedad)]:
            if col in X_new.columns:
                X_new[col] = val

        pred  = rf_model.predict(X_new)[0]
        proba = rf_model.predict_proba(X_new)[0]
        color = COLORES_SEG.get(pred, "#6366f1")

        st.markdown(f"""
        <div style="background:{color}22;border:2px solid {color};border-radius:12px;
                    padding:1.5rem;margin-top:1rem;text-align:center">
            <div style="font-size:13px;color:{color};text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:8px">Segmento predicho</div>
            <div style="font-size:2.5rem;font-weight:700;color:{color}">{pred}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probabilidad por segmento:**")
        clases = rf_model.classes_
        for cls, prob in sorted(zip(clases, proba), key=lambda x: -x[1]):
            col_c = COLORES_SEG.get(cls, "#6366f1")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0">
                <span style="width:90px;font-size:13px;color:#444">{cls}</span>
                <div style="flex:1;background:#f0f0f0;border-radius:4px;height:10px">
                    <div style="width:{prob*100:.1f}%;background:{col_c};height:10px;border-radius:4px"></div>
                </div>
                <span style="width:45px;text-align:right;font-size:13px;font-weight:600">{prob:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# COMPARACIÓN
# ══════════════════════════════════════════════════════════════
elif seccion == "⚖️ Comparación":
    st.title("⚖️ Comparación · Naturales vs Jurídicos")
    st.markdown("---")

    st.markdown("### 📉 Detalle técnico · Método del codo")
    st.markdown("Justificación de k=3 para ambos tipos de cliente.")

    col1, col2 = st.columns(2)
    for col, tipo_key in zip([col1, col2], ["NATURAL", "JURIDICO"]):
        _, X_sc, _ = calcular_clusters(tipo_key)
        color_tipo = "mediumslateblue" if tipo_key == "NATURAL" else "darkorange"
        wcss, results, previous = [], [], None
        for k in range(1, 10):
            km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=20)
            km.fit(X_sc)
            inertia   = km.inertia_
            reduccion = None if previous is None else round(previous - inertia, 1)
            wcss.append(inertia)
            results.append({"k": k, "Inercia": round(inertia, 1), "Reducción": reduccion})
            previous  = inertia
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 10)), y=wcss, mode="lines+markers",
            marker=dict(size=8, color=color_tipo),
            line=dict(color=color_tipo, width=2), name=tipo_key
        ))
        fig.add_vline(x=3, line_dash="dash", line_color="red",
                      annotation_text="k=3", annotation_position="top right")
        fig.update_layout(
            title=f"Codo · {tipo_key.title()}",
            xaxis_title="k", yaxis_title="Inercia",
            template="simple_white", height=350
        )
        col.plotly_chart(fig, use_container_width=True)
        col.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
