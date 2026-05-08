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

# ── Funciones de segmentación ──────────────────────────────
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

# ── Procesamiento base ─────────────────────────────────────
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

# ── Pipeline clustering ────────────────────────────────────
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

# ── Pipeline predicción ────────────────────────────────────
@st.cache_data(show_spinner="Entrenando modelos...")
def calcular_metricas(tipo_key):
    d = df[df["TIPO_CLIENTE"] == tipo_key].copy()

    if tipo_key == "NATURAL":
        d["TIENE_DEPTO"]      = d["DEPARTAMENTO"].notna().astype(int)
        d["TIENE_ANTIGUEDAD"] = d["ANTIGUEDAD"].notna().astype(int)
        d["DEPARTAMENTO"]     = d["DEPARTAMENTO"].fillna("NO_APLICA")
        d["ANTIGUEDAD"]       = d["ANTIGUEDAD"].fillna("NO_APLICA")
        d = d.drop(columns=["EMPRESASUNICAS_CONSULT"], errors="ignore")
        d["segmento_final"] = d["TOTAL_VENTAS"].apply(segmentar_nat)
        y = d["segmento_final"]
        numericas   = ["NUM_COMPRAS","NUM_CONSULTAS","DIASCLIENTE",
                       "PROMEDIO_VENTA","CLIENTEPORCAMPAÑAEMAIL",
                       "TIENE_DEPTO","TIENE_ANTIGUEDAD"]
        categoricas = ["CANAL_REGISTRO","DEPARTAMENTO","ANTIGUEDAD",
                       "DESC_SECTOR","ESTADO"]
    else:
        d = d.drop(columns=["EMPRESASUNICAS_CONSULT"], errors="ignore")
        d["segmento_finaljur"] = d["TOTAL_VENTAS"].apply(segmentar_jur)
        y = d["segmento_finaljur"]
        numericas   = ["NUM_COMPRAS","NUM_CONSULTAS","DIASCLIENTE",
                       "PROMEDIO_VENTA","CLIENTEPORCAMPAÑAEMAIL"]
        categoricas = ["CANAL_REGISTRO","DEPARTAMENTO","ANTIGUEDAD",
                       "DESC_SECTOR","ESTADO","TAMAÑO"]

    numericas   = [c for c in numericas   if c in d.columns]
    categoricas = [c for c in categoricas if c in d.columns]
    X = d[numericas + categoricas].copy()
    X_encoded = pd.get_dummies(X, drop_first=True).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    importancias = pd.Series(
        rf.feature_importances_, index=X_encoded.columns
    ).sort_values(ascending=False).head(10)

    # Logistic Regression
    scaler2    = StandardScaler()
    X_train_sc = scaler2.fit_transform(X_train)
    X_test_sc  = scaler2.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

    return rf, lr, scaler2, X_encoded.columns.tolist(), report_rf, report_lr, importancias

NOMBRES = {0: "Ocasionales", 1: "Recurrentes", 2: "Intensivos"}
COLORES = {"Ocasionales": "#6366f1", "Recurrentes": "#10b981", "Intensivos": "#f59e0b"}

# ── Configuración ──────────────────────────────────────────
st.set_page_config(
    page_title="Segmentación Clientes B2B · Colombia",
    page_icon="📊", layout="wide"
)
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Navegación")
seccion = st.sidebar.radio("", [
    "🏠 Inicio", "📊 Segmentación", "🔮 Predicción", "⚖️ Comparación"
])

# ══════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════
if seccion == "🏠 Inicio":
    st.title("📊 Segmentación de Clientes B2B")
    st.subheader("Análisis de recurrencia en el sector de información empresarial · Colombia")
    st.markdown("---")
    total = len(df)
    n_nat = (df["TIPO_CLIENTE"] == "NATURAL").sum()
    n_jur = (df["TIPO_CLIENTE"] == "JURIDICO").sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total clientes",     f"{total:,}")
    k2.metric("Clientes Naturales", f"{n_nat:,}", f"{n_nat/total*100:.1f}%")
    k3.metric("Clientes Jurídicos", f"{n_jur:,}", f"{n_jur/total*100:.1f}%")
    k4.metric("Modelos evaluados",  "2")
    st.markdown("---")
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
        "👥 Perfiles de clusters", "📈 Plano FM (Frecuencia vs Monto)",
        "🔵 PCA", "🔍 DBSCAN", "🧠 SOM"
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
        st.markdown("**Distribución por segmento**")
        conteo = df_seg["Segmento"].value_counts().reset_index()
        conteo.columns = ["Segmento", "Clientes"]
        fig = px.bar(conteo, x="Segmento", y="Clientes", color="Segmento",
                     color_discrete_map=COLORES, template="simple_white", text="Clientes")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("¿Cómo interpretar estos perfiles?"):
            st.markdown("""
            - **Ocasionales:** compran poco y generan bajos ingresos.
            - **Recurrentes:** frecuencia media, segmento más estable.
            - **Intensivos:** alta frecuencia, alto monto, los más valiosos.
            """)

    elif analisis == "📈 Plano FM (Frecuencia vs Monto)":
        st.subheader("Plano FM · Naturales vs Jurídicos")
        df_nat_c = df[df["TIPO_CLIENTE"] == "NATURAL"].copy()
        df_nat_c = df_nat_c[(df_nat_c["NUM_COMPRAS"] <= df_nat_c["NUM_COMPRAS"].quantile(0.95)) &
                             (df_nat_c["TOTAL_VENTAS"] <= df_nat_c["TOTAL_VENTAS"].quantile(0.95))]
        df_jur_c = df[df["TIPO_CLIENTE"] == "JURIDICO"].copy()
        df_jur_c = df_jur_c[(df_jur_c["NUM_COMPRAS"] <= df_jur_c["NUM_COMPRAS"].quantile(0.95)) &
                             (df_jur_c["TOTAL_VENTAS"] <= df_jur_c["TOTAL_VENTAS"].quantile(0.95))]
        escala = st.radio("Escala del eje Y:", ["Normal", "Logarítmica"], horizontal=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_nat_c["NUM_COMPRAS"], y=df_nat_c["TOTAL_VENTAS"],
            mode="markers", marker=dict(color="mediumslateblue", opacity=0.5, size=6, symbol="circle"),
            name="NATURAL (B2C)"))
        fig.add_trace(go.Scatter(x=df_jur_c["NUM_COMPRAS"], y=df_jur_c["TOTAL_VENTAS"],
            mode="markers", marker=dict(color="darkorange", opacity=0.7, size=7, symbol="triangle-up"),
            name="JURÍDICO (B2B)"))
        fig.update_layout(title="Plano FM · Naturales vs Jurídicos",
            xaxis_title="Frecuencia (Nº Compras)", yaxis_title="Monto Total de Ventas",
            yaxis_type="log" if escala == "Logarítmica" else "linear",
            template="simple_white", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🔵 **Naturales**")
            st.dataframe(df_nat_c[["NUM_COMPRAS","TOTAL_VENTAS"]].describe().round(2), use_container_width=True)
        with col2:
            st.markdown("🟠 **Jurídicos**")
            st.dataframe(df_jur_c[["NUM_COMPRAS","TOTAL_VENTAS"]].describe().round(2), use_container_width=True)
        with st.expander("¿Cómo interpretar el Plano FM?"):
            st.markdown("""
            - **Eje X:** frecuencia de compra. **Eje Y:** monto total.
            - Clientes esquina superior derecha → más valiosos.
            - Escala logarítmica ayuda cuando hay grandes diferencias.
            """)

    elif analisis == "🔵 PCA":
        st.subheader(f"PCA · {tipo_key.title()}")
        fig = px.scatter(df_seg, x="PC1", y="PC2", color="Segmento",
            color_discrete_map=COLORES, opacity=0.7, template="simple_white",
            title=f"Proyección PCA · {tipo_key.title()}",
            hover_data={"PC1": False, "PC2": False, "TOTAL_VENTAS": ":,.0f", "NUM_COMPRAS": True})
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.add_annotation(
            text=f"PC1 explica {var_exp[0]*100:.1f}% · PC2 explica {var_exp[1]*100:.1f}% de la varianza",
            xref="paper", yref="paper", x=0, y=-0.12, showarrow=False,
            font=dict(size=11, color="#888"))
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Varianza explicada PC1", f"{var_exp[0]*100:.1f}%")
        col2.metric("Varianza explicada PC2", f"{var_exp[1]*100:.1f}%")
        st.info(f"Entre PC1 y PC2 se explica el **{(var_exp[0]+var_exp[1])*100:.1f}%** de la varianza total.")
        with st.expander("¿Cómo interpretar el PCA?"):
            st.markdown("""
            - Cada punto es un cliente. El color indica su segmento.
            - Grupos bien separados confirman clusters distintos.
            - Mayor varianza explicada = representación más fiel.
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
        with st.expander("¿Cómo interpretar el DBSCAN?"):
            st.markdown("""
            - Agrupa clientes por densidad sin definir k de antemano.
            - **Ruido:** clientes atípicos que no encajan en ningún grupo.
            - Valida los clusters de K-Means.
            """)

    elif analisis == "🧠 SOM":
        st.subheader(f"SOM - U-Matrix · {tipo_key.title()}")
        with st.spinner("Entrenando SOM..."):
            som = MiniSom(x=6, y=6, input_len=X_scaled.shape[1],
                          sigma=1.0, learning_rate=0.5, random_seed=42)
            som.random_weights_init(X_scaled)
            som.train_random(X_scaled, 1000)
        fig = go.Figure(data=go.Heatmap(
            z=som.distance_map(), colorscale="RdYlBu_r", zsmooth="best",
            colorbar=dict(title="Distancia entre nodos")))
        fig.update_layout(title=f"SOM - U-Matrix · {tipo_key.title()}",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=500, template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("¿Cómo interpretar la U-Matrix del SOM?"):
            st.markdown("""
            - 🔵 **Colores fríos (azul)** → zonas densas = clústeres.
            - 🔴 **Colores cálidos (rojo/amarillo)** → fronteras entre segmentos.
            - Validación visual independiente de K-Means y DBSCAN.
            """)

# ══════════════════════════════════════════════════════════════
# PREDICCIÓN
# ══════════════════════════════════════════════════════════════
elif seccion == "🔮 Predicción":
    st.title("🔮 Predicción de Segmento")
    st.markdown("---")
    tipo = st.radio("Selecciona el tipo de cliente:",
                    ["🔵 Naturales", "🟣 Jurídicos"], horizontal=True)
    tipo_key   = "NATURAL" if "Naturales" in tipo else "JURIDICO"
    color_tipo = "mediumslateblue" if tipo_key == "NATURAL" else "darkorange"

    rf_model, lr_model, scaler2, feature_cols, report_rf, report_lr, importancias = calcular_metricas(tipo_key)

    st.markdown("### Comparación de modelos")
    acc_rf = report_rf["accuracy"]
    acc_lr = report_lr["accuracy"]
    col1, col2 = st.columns(2)
    col1.metric("🌲 Random Forest · Precisión global", f"{acc_rf:.0%}",
                delta="Modelo recomendado" if acc_rf > acc_lr else None)
    col2.metric("📈 Regresión Logística · Precisión global", f"{acc_lr:.0%}")
    with st.expander("¿Por qué Random Forest es mejor?"):
        st.markdown("""
        - **Random Forest** captura relaciones no lineales, ideal para segmentación.
        - **Regresión Logística** asume relaciones lineales, limitando su desempeño.
        - La diferencia es especialmente notable en los segmentos ALTO y VIP.
        """)
    st.markdown("---")

    modelo_sel = st.selectbox("Ver detalle de métricas por segmento:",
                               ["🌲 Random Forest", "📈 Regresión Logística"])
    report_sel = report_rf if "Random" in modelo_sel else report_lr
    segmentos  = ["MUY_BAJO", "BAJO", "MEDIO", "ALTO", "VIP"]
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

    st.markdown("### Variables más importantes · Random Forest")
    fig_imp = go.Figure(go.Bar(
        x=importancias.values, y=importancias.index,
        orientation="h", marker=dict(color=color_tipo)
    ))
    fig_imp.update_layout(title=f"Top 10 variables · {tipo_key.title()}",
        xaxis_title="Importancia", yaxis=dict(autorange="reversed"),
        template="simple_white", height=400)
    st.plotly_chart(fig_imp, use_container_width=True)
    with st.expander("¿Cómo usar estas variables para tomar decisiones?"):
        st.markdown("""
        - **PROMEDIO_VENTA alto** → cliente de alto valor. Priorizar fidelización.
        - **NUM_COMPRAS alto** → cliente recurrente. Ideal para programas de lealtad.
        - **NUM_CONSULTAS alto** → cliente activo. Oportunidad para ampliar oferta.

        **Estrategia por segmento:**
        - 🔴 **MUY_BAJO / BAJO:** campañas de activación y reenganche.
        - 🟡 **MEDIO:** incentivos para aumentar frecuencia de compra.
        - 🟢 **ALTO / VIP:** atención prioritaria y gestor dedicado.
        """)
    st.markdown("---")

    st.markdown("### 🔍 Predice el segmento de un cliente nuevo")

    DEPARTAMENTOS = ["BOGOTA","CUNDINAMARCA","ANTIOQUIA","VALLE","ATLANTICO",
                     "SANTANDER","BOLIVAR","TOLIMA","CALDAS","RISARALDA",
                     "QUINDIO","BOYACA","HUILA","NARINO","CAUCA","META",
                     "CESAR","MAGDALENA","CORDOBA","SUCRE","NORTE SANTANDER",
                     "LA GUAJIRA","CASANARE","PUTUMAYO","CAQUETA",
                     "SAN ANDRES","FUERA DEL PAIS","NO_APLICA"]

    SECTORES = [
        "COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACIÓN DE VEHÍCULOS AUTOMOTORES Y MOTOCICLETAS",
        "ACTIVIDADES PROFESIONALES, CIENTÍFICAS Y TÉCNICAS",
        "ACTIVIDADES FINANCIERAS Y DE SEGUROS",
        "INDUSTRIAS MANUFACTURERAS",
        "CONSTRUCCIÓN",
        "TRANSPORTE Y ALMACENAMIENTO",
        "INFORMACIÓN Y COMUNICACIONES",
        "ACTIVIDADES DE ATENCIÓN DE LA SALUD HUMANA Y DE ASISTENCIA SOCIAL",
        "EDUCACIÓN",
        "ACTIVIDADES INMOBILIARIAS",
        "ACTIVIDADES DE SERVICIOS ADMINISTRATIVOS Y DE APOYO",
        "AGRICULTURA, GANADERÍA, CAZA, SILVICULTURA Y PESCA",
        "ALOJAMIENTO Y SERVICIOS DE COMIDA",
        "OTRAS ACTIVIDADES DE SERVICIOS",
        "ADMINISTRACIÓN PÚBLICA Y DEFENSA; PLANES DE SEGURIDAD SOCIAL DE AFILIACIÓN OBLIGATORIA",
        "EXPLOTACIÓN DE MINAS Y CANTERAS",
        "SUMINISTRO DE ELECTRICIDAD, GAS, VAPOR Y AIRE ACONDICIONADO",
        "DISTRIBUCIÓN DE AGUA; EVACUACIÓN Y TRATAMIENTO DE AGUAS RESIDUALES, GESTIÓN DE DESECHOS Y ACTIVIDADES DE SANEAMIENTO AMBIENTAL",
        "NOSECTOR"
    ]

    ANTIGUEDADES = ["Menos de 3 Meses","De 3 a 18 Meses","De 3 a 5 Años",
                    "De 5 a 10 Años","Más de 10 Años","SIN FECHA DE CONSTITUCION","NO_APLICA"]

    c1, c2 = st.columns(2)
    with c1:
        promedio_venta  = st.number_input("Promedio por venta (COP)",    min_value=0.0, value=25.0,  step=5.0)
        num_compras     = st.number_input("Número de compras",            min_value=0,   value=2,     step=1)
        num_consultas   = st.number_input("Número de consultas",          min_value=0,   value=5,     step=1)
        diascliente     = st.number_input("Días como cliente",            min_value=0,   value=365,   step=30)
        email_campana   = st.selectbox("¿Cliente por campaña email?",     ["No (0)","Sí (1)"])
    with c2:
        canal           = st.selectbox("Canal de registro",               ["WEB","SEM","Directorios","Otro"])
        departamento    = st.selectbox("Departamento",                     DEPARTAMENTOS)
        antiguedad      = st.selectbox("Antigüedad",                       ANTIGUEDADES)
        sector          = st.selectbox("Sector económico",                 SECTORES)
        estado          = st.selectbox("Estado",                           ["ACTIVA","INACTIVA","EXTINGUIDA","INSOLVENTE","VIVA"])
        if tipo_key == "JURIDICO":
            tamanio     = st.selectbox("Tamaño empresa",                   ["MICRO","PEQUEÑA","MEDIANA","GRANDE","SIN DETERMINAR"])

    COLORES_SEG = {"MUY_BAJO":"#94a3b8","BAJO":"#60a5fa",
                   "MEDIO":"#34d399","ALTO":"#f59e0b","VIP":"#ef4444"}

    if st.button("🔮 Predecir segmento"):
        # Construir diccionario con los valores ingresados
        datos = {
            "NUM_COMPRAS":            num_compras,
            "NUM_CONSULTAS":          num_consultas,
            "DIASCLIENTE":            diascliente,
            "PROMEDIO_VENTA":         promedio_venta,
            "CLIENTEPORCAMPAÑAEMAIL": 1 if "Sí" in email_campana else 0,
            "CANAL_REGISTRO":         canal if canal != "Otro" else "Directorios",
            "DEPARTAMENTO":           departamento,
            "ANTIGUEDAD":             antiguedad,
            "DESC_SECTOR":            sector,
            "ESTADO":                 estado,
        }
        if tipo_key == "NATURAL":
            datos["TIENE_DEPTO"]      = 0 if departamento == "NO_APLICA" else 1
            datos["TIENE_ANTIGUEDAD"] = 0 if antiguedad   == "NO_APLICA" else 1
        else:
            datos["TAMAÑO"] = tamanio

        # One-Hot igual al notebook
        df_nuevo = pd.DataFrame([datos])
        df_nuevo_enc = pd.get_dummies(df_nuevo, drop_first=True).astype(int)
        df_nuevo_enc = df_nuevo_enc.reindex(columns=feature_cols, fill_value=0)

        pred  = rf_model.predict(df_nuevo_enc)[0]
        proba = rf_model.predict_proba(df_nuevo_enc)[0]
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
        for cls, prob in sorted(zip(rf_model.classes_, proba), key=lambda x: -x[1]):
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
        fig.add_trace(go.Scatter(x=list(range(1, 10)), y=wcss, mode="lines+markers",
            marker=dict(size=8, color=color_tipo), line=dict(color=color_tipo, width=2)))
        fig.add_vline(x=3, line_dash="dash", line_color="red",
                      annotation_text="k=3", annotation_position="top right")
        fig.update_layout(title=f"Codo · {tipo_key.title()}",
            xaxis_title="k", yaxis_title="Inercia",
            template="simple_white", height=350)
        col.plotly_chart(fig, use_container_width=True)
        col.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
