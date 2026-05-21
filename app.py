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

# ── Recomendaciones de negocio por segmento ────────────────
RECOMENDACIONES = {
    "NATURAL": {
        "Ocasionales": {
            "color": "#6366f1",
            "icono": "🔵",
            "perfil": "Clientes con 1-2 compras, bajo ticket promedio y pocas consultas. Muchos sin departamento registrado (persona física).",
            "acciones": [
                "📧 Campaña de reactivación por email — ya tienen `CLIENTEPORCAMPAÑAEMAIL` registrado",
                "🎯 Oferta de segunda compra con descuento del 10-15%",
                "📞 Contacto comercial proactivo para entender sus necesidades",
                "📊 Monitorear si consultan pero no compran — son leads calientes",
            ]
        },
        "Recurrentes": {
            "color": "#10b981",
            "icono": "🟢",
            "perfil": "Clientes con frecuencia de compra media, ticket moderado y nivel de consultas activo. Segmento más estable y predecible.",
            "acciones": [
                "🏆 Programa de fidelización basado en número de compras acumuladas",
                "🔔 Alertas personalizadas de nuevos productos según su sector económico",
                "💰 Descuentos por volumen para incentivar mayor ticket por compra",
                "📈 Objetivo: moverlos hacia el segmento Intensivo en 6 meses",
            ]
        },
        "Intensivos": {
            "color": "#f59e0b",
            "icono": "🟡",
            "perfil": "Alto PROMEDIO_VENTA, muchas compras y muchas consultas. Son pocos pero generan la mayor parte del ingreso.",
            "acciones": [
                "👤 Asignar gestor de cuenta dedicado — son clientes estratégicos",
                "🌟 Acceso prioritario a nuevos productos e información exclusiva",
                "🤝 Reuniones periódicas para entender su evolución de negocio",
                "🔒 Contrato de fidelización a largo plazo con condiciones preferenciales",
            ]
        }
    },
    "JURIDICO": {
        "Ocasionales": {
            "color": "#6366f1",
            "icono": "🔵",
            "perfil": "Empresas con 1-2 compras y bajo volumen. Tienen datos completos (departamento, antigüedad, tamaño) — se puede personalizar mucho la estrategia.",
            "acciones": [
                "🏢 Visita comercial presencial — son empresas, el contacto directo funciona mejor",
                "📋 Propuesta personalizada según su sector y tamaño empresarial",
                "🔍 Analizar si consultan mucho pero no compran — puede ser barrera de precio",
                "📧 Campaña de nurturing B2B por sector económico",
            ]
        },
        "Recurrentes": {
            "color": "#10b981",
            "icono": "🟢",
            "perfil": "Empresas con compras periódicas, ticket medio y buen nivel de consultas. Segmento con mayor potencial de crecimiento.",
            "acciones": [
                "📄 Contrato marco anual con condiciones fijas — reduce fricción de compra",
                "💼 Descuentos por volumen acumulado trimestral",
                "📊 Reporte periódico de uso y valor generado — refuerza la relación",
                "🎯 Objetivo: aumentar ticket promedio mediante venta cruzada por sector",
            ]
        },
        "Intensivos": {
            "color": "#f59e0b",
            "icono": "🟡",
            "perfil": "Empresas con alto PROMEDIO_VENTA, muchas compras y consulta de muchas empresas únicas. Son los clientes más valiosos y los más difíciles de reemplazar.",
            "acciones": [
                "👔 Account Manager exclusivo con SLA de respuesta en menos de 2 horas",
                "🔗 Integración de sistemas o API para automatizar sus consultas",
                "📑 Acuerdo estratégico de largo plazo con revisión anual de condiciones",
                "🏆 Programa VIP con acceso anticipado a nuevas bases de datos y sectores",
            ]
        }
    }
}

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
    numericas   = ["NUM_COMPRAS","NUM_CONSULTAS","EMPRESASUNICAS_CONSULT",
                   "DIASCLIENTE","PROMEDIO_VENTA","CLIENTEPORCAMPAÑAEMAIL"]
    categoricas = ["CANAL_REGISTRO","DEPARTAMENTO","ANTIGUEDAD",
                   "DESC_SECTOR","ESTADO","TAMAÑO"]
    if tipo_key == "NATURAL":
        d["DEPARTAMENTO"] = d["DEPARTAMENTO"].fillna("NO_APLICA")
        d["ANTIGUEDAD"]   = d["ANTIGUEDAD"].fillna("NO_APLICA")
        d["TAMAÑO"]       = d["TAMAÑO"].fillna("NO_APLICA")
        d["segmento_final"] = d["TOTAL_VENTAS"].apply(segmentar_nat)
        y = d["segmento_final"]
    else:
        d["segmento_finaljur"] = d["TOTAL_VENTAS"].apply(segmentar_jur)
        y = d["segmento_finaljur"]
    numericas   = [c for c in numericas   if c in d.columns]
    categoricas = [c for c in categoricas if c in d.columns]
    X = d[numericas + categoricas].copy()
    X_encoded = pd.get_dummies(X, drop_first=True).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    importancias = pd.Series(
        rf.feature_importances_, index=X_encoded.columns
    ).sort_values(ascending=False).head(10)
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
    "🏠 Inicio", "📊 Segmentación", "🔮 Predicción", "⚖️ Comparación", "📂 Cargar datos"
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
                rec = RECOMENDACIONES[tipo_key][seg]
                col.markdown(f"**{rec['icono']} {seg}**")
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

        # ── Recomendaciones de negocio ─────────────────────
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones de negocio por segmento")
        for seg in ["Ocasionales", "Recurrentes", "Intensivos"]:
            rec = RECOMENDACIONES[tipo_key][seg]
            with st.expander(f"{rec['icono']} **{seg}** — {rec['perfil']}"):
                st.markdown("**Acciones recomendadas:**")
                for accion in rec["acciones"]:
                    st.markdown(f"- {accion}")

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

        st.markdown("### 💡 Lectura de negocio · PCA")
        rec_pca = RECOMENDACIONES[tipo_key]
        col1, col2, col3 = st.columns(3)
        for col, seg in zip([col1, col2, col3], ["Ocasionales", "Recurrentes", "Intensivos"]):
            rec = rec_pca[seg]
            with col:
                st.markdown(f"**{rec['icono']} {seg}**")
                st.markdown(f"<small style='color:#666'>{rec['perfil']}</small>", unsafe_allow_html=True)
                st.markdown("**Acción clave:**")
                st.info(rec["acciones"][0])

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
            - **Ruido:** clientes atípicos que no encajan en ningún grupo — merecen análisis individual.
            - Si DBSCAN encuentra los mismos grupos que K-Means, la segmentación es sólida.
            """)
        pct_ruido = n_ruido / len(df_seg) * 100
        if n_ruido > 0:
            if pct_ruido > 5:
                st.warning(f"⚠️ {n_ruido} clientes ({pct_ruido:.1f}%) clasificados como ruido.")
            else:
                st.success(f"✅ Solo {n_ruido} clientes ({pct_ruido:.1f}%) como ruido — segmentación limpia.")

        st.markdown("### 💡 Lectura de negocio · DBSCAN")
        if tipo_key == "NATURAL":
            st.markdown(f"""
**Clientes en los clusters definidos ({len(df_seg) - n_ruido:,}):**
- 🔵 Los clusters bien definidos confirman que los 3 segmentos tienen comportamientos claramente distintos entre sí.
- El algoritmo los agrupa por densidad sin forzar — si coincide con K-Means, la segmentación es sólida.

**Clientes en Ruido ({n_ruido:,} · {pct_ruido:.1f}%):**
- Son personas naturales con comportamiento atípico — compran mucho más o con una frecuencia inusual.
- **Acción:** revisarlos individualmente. Pueden ser empresarios clasificados como persona física o clientes con potencial VIP no detectado.
- Priorizar contacto comercial directo con este grupo.
            """)
        else:
            st.markdown(f"""
**Clientes en los clusters definidos ({len(df_seg) - n_ruido:,}):**
- Los clusters reflejan empresas con patrones de compra consistentes dentro de su segmento.
- La separación por densidad valida que Ocasionales, Recurrentes e Intensivos son grupos reales, no artificiales.

**Clientes en Ruido ({n_ruido:,} · {pct_ruido:.1f}%):**
- Empresas con comportamiento fuera de lo común — pueden ser grandes corporaciones, holdings o empresas en proceso de cambio.
- **Acción:** análisis individual por equipo comercial. Verificar si son clientes estratégicos mal clasificados o casos con datos incompletos.
- Alta probabilidad de que algunos sean candidatos a segmento VIP en la predicción.
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
            - 🔵 **Colores fríos (azul)** → zonas densas = clústeres bien definidos.
            - 🔴 **Colores cálidos (rojo/amarillo)** → fronteras naturales entre segmentos.
            - Zonas azules claramente separadas confirman que los 3 segmentos son distintos.
            - Es una validación visual independiente de K-Means y DBSCAN.
            """)

        st.markdown("### 💡 Lectura de negocio · SOM")
        if tipo_key == "NATURAL":
            st.markdown("""
**¿Qué nos dice el mapa sobre los clientes naturales?**

- 🔵 **Zona densa izquierda (azul intensa):** concentra la mayoría de clientes — son los **Ocasionales y Recurrentes**. Gran volumen, comportamiento predecible. El foco aquí es activación y fidelización masiva.

- 🔴 **Zona cálida central (naranja/amarillo):** frontera entre segmentos — clientes en transición. Son los más interesantes comercialmente: con un pequeño empujón pueden subir de segmento.
  - **Acción:** campaña dirigida específicamente a este grupo con oferta de upgrade.

- 🔵 **Zona densa derecha (azul):** clientes **Intensivos** — pocos pero con alto valor. El mapa los separa claramente del resto, lo que confirma que son un grupo diferente que necesita estrategia propia.
  - **Acción:** atención personalizada, no campaña masiva.
            """)
        else:
            st.markdown("""
**¿Qué nos dice el mapa sobre los clientes jurídicos?**

- 🔵 **Zonas azules densas:** agrupan empresas con comportamiento consistente — **Ocasionales con bajo volumen** y **Recurrentes estables**. Son la base del negocio B2B.
  - **Acción:** contratos marco y descuentos por volumen para consolidar la relación.

- 🔴 **Zonas cálidas (fronteras):** empresas en transición entre segmentos. En el contexto jurídico esto suele ocurrir cuando una empresa está creciendo o cambiando de sector.
  - **Acción:** visita comercial para entender su momento de negocio y ajustar la propuesta.

- 🔵 **Zona aislada (azul separada):** empresas **Intensivas** — corporaciones o grupos empresariales con alta frecuencia y ticket elevado. El SOM los separa del resto de forma natural.
  - **Acción:** account manager dedicado y acuerdo estratégico de largo plazo.
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

    if tipo_key == "NATURAL":
        DEPARTAMENTOS = ["NO_APLICA","ANTIOQUIA","ARAUCA","ATLANTICO","BOGOTA",
                         "BOLIVAR","BOYACA","CALDAS","CAQUETA","CASANARE","CAUCA",
                         "CESAR","CHOCO","CORDOBA","CUNDINAMARCA","GUAVIARE","HUILA",
                         "LA GUAJIRA","MAGDALENA","META","NARINO","NORTE SANTANDER",
                         "PUTUMAYO","QUINDIO","RISARALDA","SAN ANDRES","SANTANDER",
                         "SUCRE","TOLIMA","VALLE"]
        SECTORES     = ["NOSECTOR",
            "ACTIVIDADES DE ATENCIÓN DE LA SALUD HUMANA Y DE ASISTENCIA SOCIAL",
            "ACTIVIDADES DE LOS HOGARES INDIVIDUALES EN CALIDAD DE EMPLEADORES; ACTIVIDADES NO DIFERENCIADAS DE LOS HOGARES INDIVIDUALES COMO PRODUCTORES DE BIENES Y SERVICIOS PARA USO PROPIO",
            "ACTIVIDADES DE SERVICIOS ADMINISTRATIVOS Y DE APOYO",
            "ACTIVIDADES FINANCIERAS Y DE SEGUROS","ACTIVIDADES INMOBILIARIAS",
            "ACTIVIDADES PROFESIONALES, CIENTÍFICAS Y TÉCNICAS",
            "ADMINISTRACIÓN PÚBLICA Y DEFENSA; PLANES DE SEGURIDAD SOCIAL DE AFILIACIÓN OBLIGATORIA",
            "AGRICULTURA, GANADERÍA, CAZA, SILVICULTURA Y PESCA",
            "ALOJAMIENTO Y SERVICIOS DE COMIDA","COMERCIAL / INDUSTRIAL NO DEFINIDA",
            "COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACIÓN DE VEHÍCULOS AUTOMOTORES Y MOTOCICLETAS",
            "CONSTRUCCIÓN",
            "DISTRIBUCIÓN DE AGUA; EVACUACIÓN Y TRATAMIENTO DE AGUAS RESIDUALES, GESTIÓN DE DESECHOS Y ACTIVIDADES DE SANEAMIENTO AMBIENTAL",
            "EDUCACIÓN","EXPLOTACIÓN DE MINAS Y CANTERAS","INDUSTRIAS MANUFACTURERAS",
            "INFORMACIÓN Y COMUNICACIONES","OTRAS ACTIVIDADES DE SERVICIOS",
            "TRANSPORTE Y ALMACENAMIENTO"]
        ESTADOS      = ["ACTIVA","INACTIVA","INSOLVENTE","VIVA"]
        ANTIGUEDADES = ["NO_APLICA","Menos de 3 Meses","De 3 a 18 Meses",
                        "De 3 a 5 Años","De 5 a 10 Años","Más de 10 Años",
                        "SIN FECHA DE CONSTITUCION"]
    else:
        DEPARTAMENTOS = ["ATLANTICO","BOGOTA","BOLIVAR","BOYACA","CALDAS",
                         "CAQUETA","CASANARE","CAUCA","CESAR","CORDOBA",
                         "CUNDINAMARCA","FUERA DEL PAIS","HUILA","LA GUAJIRA",
                         "MAGDALENA","META","NARINO","NORTE SANTANDER","PUTUMAYO",
                         "QUINDIO","RISARALDA","SAN ANDRES","SANTANDER","SUCRE",
                         "TOLIMA","VALLE"]
        SECTORES     = [
            "ACTIVIDADES DE ATENCIÓN DE LA SALUD HUMANA Y DE ASISTENCIA SOCIAL",
            "ACTIVIDADES DE LOS HOGARES INDIVIDUALES EN CALIDAD DE EMPLEADORES; ACTIVIDADES NO DIFERENCIADAS DE LOS HOGARES INDIVIDUALES COMO PRODUCTORES DE BIENES Y SERVICIOS PARA USO PROPIO",
            "ACTIVIDADES DE SERVICIOS ADMINISTRATIVOS Y DE APOYO",
            "ACTIVIDADES FINANCIERAS Y DE SEGUROS","ACTIVIDADES INMOBILIARIAS",
            "ACTIVIDADES PROFESIONALES, CIENTÍFICAS Y TÉCNICAS",
            "ADMINISTRACIÓN PÚBLICA Y DEFENSA; PLANES DE SEGURIDAD SOCIAL DE AFILIACIÓN OBLIGATORIA",
            "AGRICULTURA, GANADERÍA, CAZA, SILVICULTURA Y PESCA",
            "ALOJAMIENTO Y SERVICIOS DE COMIDA",
            "COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACIÓN DE VEHÍCULOS AUTOMOTORES Y MOTOCICLETAS",
            "CONSTRUCCIÓN",
            "DISTRIBUCIÓN DE AGUA; EVACUACIÓN Y TRATAMIENTO DE AGUAS RESIDUALES, GESTIÓN DE DESECHOS Y ACTIVIDADES DE SANEAMIENTO AMBIENTAL",
            "EDUCACIÓN","EXPLOTACIÓN DE MINAS Y CANTERAS","INDUSTRIAS MANUFACTURERAS",
            "INFORMACIÓN Y COMUNICACIONES","OTRAS ACTIVIDADES DE SERVICIOS",
            "SUMINISTRO DE ELECTRICIDAD, GAS, VAPOR Y AIRE ACONDICIONADO",
            "TRANSPORTE Y ALMACENAMIENTO"]
        ESTADOS      = ["ACTIVA","INACTIVA","INSOLVENTE","EXTINGUIDA"]
        ANTIGUEDADES = ["De 3 a 18 Meses","De 3 a 5 Años","De 5 a 10 Años",
                        "Más de 10 Años","SIN FECHA DE CONSTITUCION"]

    c1, c2 = st.columns(2)
    with c1:
        promedio_venta = st.number_input("Promedio por venta (COP)",        min_value=0.0, value=25.0, step=5.0)
        num_compras    = st.number_input("Número de compras",                min_value=0,   value=2,   step=1)
        num_consultas  = st.number_input("Número de consultas",              min_value=0,   value=5,   step=1)
        emp_unicas     = st.number_input("Empresas únicas consultadas",      min_value=0,   value=3,   step=1)
        diascliente    = st.number_input("Días como cliente",                min_value=0,   value=365, step=30)
        email_campana  = st.selectbox("¿Cliente por campaña email?",         ["No (0)","Sí (1)"])
    with c2:
        canal          = st.selectbox("Canal de registro",                   ["WEB","SEM","Directorios","Otro"])
        departamento   = st.selectbox("Departamento",                        DEPARTAMENTOS)
        antiguedad     = st.selectbox("Antigüedad",                          ANTIGUEDADES)
        sector         = st.selectbox("Sector económico",                    SECTORES)
        estado         = st.selectbox("Estado",                              ESTADOS)
        tamanio        = st.selectbox("Tamaño empresa",                      ["NO_APLICA","MICRO","PEQUEÑA","MEDIANA","GRANDE","SIN DETERMINAR"])

    COLORES_SEG = {"MUY_BAJO":"#94a3b8","BAJO":"#60a5fa",
                   "MEDIO":"#34d399","ALTO":"#f59e0b","VIP":"#ef4444"}

    REC_PRED = {
        "MUY_BAJO": "🔴 Campaña de reactivación urgente. Considerar contacto directo para entender barreras de compra.",
        "BAJO":     "🟠 Cliente con potencial. Oferta personalizada de segunda compra y seguimiento comercial.",
        "MEDIO":    "🟡 Cliente estable. Incentivos para aumentar frecuencia — descuentos por volumen o programa de puntos.",
        "ALTO":     "🟢 Cliente valioso. Atención preferencial, acceso anticipado a nuevos productos y fidelización activa.",
        "VIP":      "⭐ Cliente estratégico. Gestor dedicado, condiciones exclusivas y relación de largo plazo."
    }

    if st.button("🔮 Predecir segmento"):
        X_new = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)
        X_new["PROMEDIO_VENTA"]         = promedio_venta
        X_new["NUM_COMPRAS"]            = num_compras
        X_new["NUM_CONSULTAS"]          = num_consultas
        X_new["DIASCLIENTE"]            = diascliente
        X_new["CLIENTEPORCAMPAÑAEMAIL"] = 1 if "Sí" in email_campana else 0
        if "EMPRESASUNICAS_CONSULT" in X_new.columns:
            X_new["EMPRESASUNICAS_CONSULT"] = emp_unicas
        if canal == "WEB" and "CANAL_REGISTRO_WEB" in X_new.columns:
            X_new["CANAL_REGISTRO_WEB"] = 1
        elif canal == "SEM" and "CANAL_REGISTRO_SEM" in X_new.columns:
            X_new["CANAL_REGISTRO_SEM"] = 1
        col_dep = f"DEPARTAMENTO_{departamento}"
        if col_dep in X_new.columns: X_new[col_dep] = 1
        col_ant = f"ANTIGUEDAD_{antiguedad}"
        if col_ant in X_new.columns: X_new[col_ant] = 1
        col_sec = f"DESC_SECTOR_{sector}"
        if col_sec in X_new.columns: X_new[col_sec] = 1
        col_est = f"ESTADO_{estado}"
        if col_est in X_new.columns: X_new[col_est] = 1
        col_tam = f"TAMAÑO_{tamanio}"
        if col_tam in X_new.columns: X_new[col_tam] = 1

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

        st.markdown("---")
        st.markdown(f"**💡 Recomendación:** {REC_PRED.get(pred, '')}")

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

# ══════════════════════════════════════════════════════════════
# CARGAR DATOS
# ══════════════════════════════════════════════════════════════
elif seccion == "📂 Cargar datos":
    st.title("📂 Cargar datos actualizados")
    st.markdown("---")
    st.info("Sube los tres archivos con la misma estructura para analizar datos nuevos.")

    f_clientes  = st.file_uploader("CLIENTES.txt",         type=["txt","csv"])
    f_ventas    = st.file_uploader("VENTAS.txt",           type=["txt","csv"])
    f_consultas = st.file_uploader("CONSULTAS.txt / .zip", type=["txt","csv","zip"])

    if f_clientes and f_ventas and f_consultas:
        import zipfile, io
        try:
            clientes_new = pd.read_csv(f_clientes, sep="|", encoding="latin-1")
            ventas_new   = pd.read_csv(f_ventas,   sep="|", encoding="latin-1")
            if f_consultas.name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(f_consultas.read())) as z:
                    nombre = [f for f in z.namelist() if f.endswith(".txt")][0]
                    with z.open(nombre) as f:
                        consultas_new = pd.read_csv(f, sep="|", encoding="latin-1")
            else:
                consultas_new = pd.read_csv(f_consultas, sep="|", encoding="latin-1")

            df_new = procesar_datos(clientes_new, ventas_new, consultas_new)
            total  = len(df_new)
            n_nat  = (df_new["TIPO_CLIENTE"] == "NATURAL").sum()
            n_jur  = (df_new["TIPO_CLIENTE"] == "JURIDICO").sum()

            st.success("✅ Archivos cargados correctamente")
            k1, k2, k3 = st.columns(3)
            k1.metric("Total clientes",     f"{total:,}")
            k2.metric("Clientes Naturales", f"{n_nat:,}", f"{n_nat/total*100:.1f}%")
            k3.metric("Clientes Jurídicos", f"{n_jur:,}", f"{n_jur/total*100:.1f}%")
            st.markdown("---")
            st.markdown("**Vista previa:**")
            st.dataframe(df_new.head(10), use_container_width=True)
            st.info("Para analizar estos datos completos, reemplaza los archivos en GitHub y la app se actualizará automáticamente.")
        except Exception as e:
            st.error(f"Error al procesar los archivos: {e}")
    else:
        st.markdown("""
        **Estructura requerida:**
        - **CLIENTES.txt** — separado por `|` con las mismas columnas originales
        - **VENTAS.txt** — separado por `|` con columnas `ID` e `IMPORTE`
        - **CONSULTAS.txt** — separado por `|` con columnas `IDCONSUMO` e `ID`
        """)
