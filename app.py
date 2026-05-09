import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from datetime import datetime
import subprocess

# ══════════════════════════════════════════════════════════
#  CONFIGURACIÓN Y ESTÉTICA PRO
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="Air Intel | Monitor Pro", page_icon="📡", layout="wide")

COLORS = {
    'bg':        '#020617',
    'surface':   '#0f172a',
    'glass':     'rgba(30, 41, 59, 0.7)',
    'border':    '#334155', 
    'text':      '#f8fafc',
    'subtext':   '#94a3b8',
    'accent':    '#38bdf8',
    'success':   '#10b981',
    'warning':   '#f59e0b',
    'danger':    '#ef4444'
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; color: {COLORS['text']}; }}
.stApp {{ background: {COLORS['bg']}; }}
.glass-card {{
    background: {COLORS['glass']}; backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 20px;
    padding: 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
}}
.header-pro {{
    background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 3rem; letter-spacing: -0.05em;
}}
.prediction-box {{
    border-radius: 20px; padding: 2.5rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    animation: fadeIn 0.5s ease-out;
}}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  LÓGICA DE AUTO-ENTRENAMIENTO (MANDATORY)
# ══════════════════════════════════════════════════════════
def check_and_retrain():
    dataset_path = 'augmented_air_quality.csv'
    model_path = 'air_nn_model.pkl'
    
    if os.path.exists(dataset_path) and os.path.exists(model_path):
        data_time = os.path.getmtime(dataset_path)
        model_time = os.path.getmtime(model_path)
        
        # Si el dataset es más nuevo que el modelo (Margen de 2 seg para evitar loops)
        if data_time > (model_time + 2):
            st.warning("🔄 Se detectaron nuevos datos en el dataset. Re-entrenando modelos para máxima precisión...")
            with st.spinner("Cerebro de IA actualizándose (RandomizedSearchCV en progreso)..."):
                try:
                    # Ejecutamos ambos entrenamientos
                    subprocess.run(["python", "train_air_nn.py"], check=True)
                    subprocess.run(["python", "train_air_logistic.py"], check=True)
                    st.success("✨ Modelos actualizados con éxito con los nuevos datos.")
                    st.cache_resource.clear()
                    return True
                except Exception as e:
                    st.error(f"Error en auto-entrenamiento: {e}")
    return False

# Ejecutamos el check al inicio
if check_and_retrain():
    st.rerun()

# ══════════════════════════════════════════════════════════
#  LÓGICA DE RECURSOS
# ══════════════════════════════════════════════════════════
def get_resource_fingerprint():
    files = ['air_logistic_model.pkl', 'air_nn_model.pkl', 'air_logistic_metrics.json', 'air_nn_metrics.json']
    fingerprint = []
    for f in files:
        if os.path.exists(f):
            stats = os.stat(f)
            fingerprint.append((f, stats.st_mtime, stats.st_size))
    return tuple(fingerprint)

@st.cache_resource(show_spinner="Sincronizando motores de IA...")
def load_assets(fingerprint):
    l_m = joblib.load('air_logistic_model.pkl')
    n_m = joblib.load('air_nn_model.pkl')
    s = joblib.load('air_scaler.pkl')
    with open('air_logistic_metrics.json', 'r') as f: m_l = json.load(f)
    with open('air_nn_metrics.json', 'r') as f: m_n = json.load(f)
    return l_m, n_m, s, m_l, m_n, datetime.now().strftime("%H:%M:%S")

l_model, n_model, scaler, m_log, m_nn, load_time = load_assets(get_resource_fingerprint())

PERIOD_MAP = {
    'Morning Entry': 0, '1st Period': 1, '2nd Period': 2, 'Break': 3,
    '3rd Period': 4, '4th Period': 5, 'Lunch Break': 6, '5th Period': 7,
    'End of Day': 8
}
LABEL_MAP = {0: 'Good', 1: 'Moderate', 2: 'Poor'}
FEATURE_COLS = ['school_period', 'student_count_estimated', 'co2_ppm', 'pm25_ugm3', 'temperature_c', 'humidity_pct', 'robot_x_pos', 'robot_y_pos']

# ══════════════════════════════════════════════════════════
#  VISUALIZACIÓN DE MATRIZ
# ══════════════════════════════════════════════════════════
def plot_matrix_pro(cm, labels, title):
    fig, ax = plt.subplots(figsize=(6, 5)); fig.patch.set_facecolor('none'); ax.set_facecolor('none')
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, color='white', pad=20, fontsize=14, fontweight='bold')
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks); ax.set_xticklabels(labels, color=COLORS['subtext'], rotation=45)
    ax.set_yticks(tick_marks); ax.set_yticklabels(labels, color=COLORS['subtext'])
    thresh = np.array(cm).max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i][j]
            txt_color = "white" if val > thresh else "#0f172a" 
            ax.text(j, i, format(val, 'd'), ha="center", va="center", color=txt_color, fontsize=14, fontweight='bold')
    for spine in ax.spines.values(): spine.set_edgecolor(COLORS['border'])
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════
#  SIDEBAR DASHBOARD
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8;'>Panel de Control</h2>", unsafe_allow_html=True)
    st.caption(f"🚀 Modelos cargados a las: {load_time}")
    model_choice = st.radio("Motor de Inferencia", ["Regresión Logística", "Red Neuronal"], index=1)
    act_mod = l_model if model_choice == "Regresión Logística" else n_model
    act_met = m_log if model_choice == "Regresión Logística" else m_nn
    st.markdown("---")
    col_a, col_b = st.columns(2)
    col_a.metric("Accuracy", f"{act_met['accuracy']*100:.1f}%")
    col_b.metric("F1-Score", f"{act_met['f1_score']*100:.1f}%")
    st.markdown("#### Matriz Global (Train)")
    fig_side = plot_matrix_pro(np.array(act_met['confusion_matrix']), [LABEL_MAP[c] for c in act_mod.classes_], "Rendimiento Histórico")
    st.pyplot(fig_side)
    if st.button("🔄 Sincronizar Disco", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════
#  CUERPO PRINCIPAL
# ══════════════════════════════════════════════════════════
st.markdown("<h1 class='header-pro'>Air Intel <span style='font-size:1.5rem; vertical-align:middle;'>v2.1</span></h1>", unsafe_allow_html=True)

# Session State
for key in ['period', 'students', 'co2', 'pm25', 'temp', 'hum', 'rx', 'ry']:
    if key not in st.session_state: st.session_state[key] = 'Morning Entry' if key == 'period' else 0.0

def randomize_data():
    st.session_state.period = random.choice(list(PERIOD_MAP.keys()))
    st.session_state.students = random.randint(5, 45); st.session_state.co2 = round(random.uniform(400, 1800), 1)
    st.session_state.pm25 = round(random.uniform(5, 45), 2); st.session_state.temp = round(random.uniform(18, 30), 1)
    st.session_state.hum = round(random.uniform(30, 75), 1); st.session_state.rx = round(random.uniform(0, 10), 2)
    st.session_state.ry = round(random.uniform(0, 10), 2)

tab1, tab2 = st.tabs(["🎯 Análisis Individual", "📦 Procesamiento Masivo"])

with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c_main, c_res = st.columns([2, 1])
    with c_main:
        st.markdown("### 🧬 Parámetros en Tiempo Real")
        st.button("🎲 Generar Escenario Aleatorio", on_click=randomize_data)
        ca, cb = st.columns(2)
        with ca:
            period = st.selectbox("Horario Escolar", list(PERIOD_MAP.keys()), key='period')
            students = st.number_input("Ocupación Est.", 0, 60, key='students')
            co2 = st.number_input("Nivel CO2 (ppm)", 0.0, 3000.0, key='co2')
            pm25 = st.number_input("Partículas PM2.5", 0.0, 150.0, key='pm25')
        with cb:
            temp = st.number_input("Temp. Ambiente (°C)", 0.0, 50.0, key='temp')
            hum = st.number_input("Humedad Rel. (%)", 0.0, 100.0, key='hum')
            rx = st.number_input("X-Pos Robot", 0.0, 10.0, key='rx')
            ry = st.number_input("Y-Pos Robot", 0.0, 10.0, key='ry')
        btn_run = st.button("🚀 Ejecutar Diagnóstico", type="primary", use_container_width=True)
    with c_res:
        if btn_run and co2 > 0:
            df_inf = pd.DataFrame([[PERIOD_MAP[period], students, co2, pm25, temp, hum, rx, ry]], columns=FEATURE_COLS)
            inf_scaled = scaler.transform(df_inf)
            pred = act_mod.predict(inf_scaled)[0]; prob = act_mod.predict_proba(inf_scaled)[0]; label = LABEL_MAP[pred]
            st.toast(f"Análisis IA: {label}", icon="✨")
            color = COLORS['success'] if label == 'Good' else (COLORS['warning'] if label == 'Moderate' else COLORS['danger'])
            st.markdown(f"<div class='prediction-box' style='background: {color}22; border-color: {color};'><h4 style='color:{color}; margin:0;'>ESTADO DETECTADO</h4><h1 style='color:{color}; margin:0; font-size:3.5rem;'>{label}</h1><span style='background:{color}; color:white; padding: 0.5rem 1rem; border-radius: 99px; font-weight: bold;'>CONFIANZA {prob[pred]*100:.1f}%</span></div>", unsafe_allow_html=True)
            fig_p, ax_p = plt.subplots(figsize=(4, 3)); fig_p.patch.set_facecolor('none'); ax_p.set_facecolor('none')
            ax_p.barh([LABEL_MAP[c] for c in act_mod.classes_], prob, color=[COLORS['success'], COLORS['warning'], COLORS['danger']])
            ax_p.set_xlim(0, 1); ax_p.tick_params(colors=COLORS['subtext']); st.pyplot(fig_p)
        else:
            st.markdown("<div style='text-align:center; padding:3rem; color:#64748b;'><h3>Esperando entrada...</h3><p>Cargue datos o use el modo aleatorio</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📦 Ingesta de Datos en Lote")
    up_file = st.file_uploader("Arrastre su archivo .csv aquí", type=['csv'])
    if up_file:
        df_batch = pd.read_csv(up_file)
        df_batch.columns = df_batch.columns.str.strip().str.lower()
        if st.button("⚡ Iniciar Procesamiento Masivo", use_container_width=True):
            proc = df_batch.copy()
            p_col = next((c for c in df_batch.columns if 'period' in c), 'school_period')
            proc[p_col] = proc[p_col].map(PERIOD_MAP)
            X_b = pd.DataFrame()
            for c in FEATURE_COLS:
                found = next((f for f in df_batch.columns if c in f or f in c), None)
                X_b[c] = proc[found] if found else 0
            preds = act_mod.predict(scaler.transform(X_b))
            df_batch['PREDICCIÓN'] = [LABEL_MAP[p] for p in preds]
            st.toast("Procesamiento completado", icon="✅")
            st.markdown("#### 📋 Resultados de Clasificación")
            t_col = next((c for c in df_batch.columns if 'time' in c), None)
            st.dataframe(df_batch[[t_col, 'PREDICCIÓN'] if t_col else ['PREDICCIÓN']], width='stretch')
            l_col = next((c for c in df_batch.columns if 'label' in c or 'quality' in c), None)
            if l_col:
                from sklearn.metrics import confusion_matrix as sk_cm
                st.markdown("---")
                st.markdown("#### 📊 Análisis de Precisión del Lote")
                y_t = df_batch[l_col].astype(str).str.capitalize().map({'Good': 0, 'Moderate': 1, 'Poor': 2})
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    fig_b = plot_matrix_pro(sk_cm(y_t, preds, labels=act_mod.classes_), [LABEL_MAP[c] for c in act_mod.classes_], "Validación de Lote")
                    st.pyplot(fig_b)
                with c2: st.success("Matriz generada con éxito.")
            st.download_button("📥 Exportar Reporte (.csv)", df_batch.to_csv(index=False), "air_intel_report.csv", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
