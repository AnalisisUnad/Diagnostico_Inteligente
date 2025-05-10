import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diagnóstico Médico con IA", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    .titulo {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #6c63ff;
        margin-bottom: 10px;
    }
    .subtitulo {
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .diagnostico-box {
        background-color: #e6ffe6;
        padding: 15px;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    <div class="main">
        <div class="titulo">🧠 Diagnóstico Médico Automatizado con IA</div>
        <div class="subtitulo">Seleccione los síntomas que presenta el paciente (1 = Sí, 0 = No)</div>
    </div>
""", unsafe_allow_html=True)

nombres_sintomas = [
    "Tos", "Fiebre", "Dolor de cabeza", "Fatiga", "Dolor muscular",
    "Dolor de garganta", "Congestión nasal", "Dificultad para respirar", "Pérdida de apetito"
]

inputs = []
with st.form("formulario_sintomas"):
    for sintoma in nombres_sintomas:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{sintoma}:**")
        with col2:
            valor = st.radio(
                "", [1, 0],
                horizontal=True,
                label_visibility="collapsed",
                key=f"radio_{sintoma}"
            )
            inputs.append(valor)
    enviado = st.form_submit_button("🔍 Predecir diagnóstico")

sintomas = [
    [1, 1, 1, 1, 1, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 1], [0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 0, 0, 0, 1], [1, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 1, 0, 0]
]

diagnosticos = [
    "Gripe", "Resfriado", "Alergia", "Neumonía", "Sinusitis", "Bronquitis", "COVID-19",
    "Faringitis", "Asma", "Tuberculosis", "Bronquiectasia", "Laringitis", "Otitis", "EPOC",
    "Neumonía bacteriana", "Neumonía viral", "Faringoamigdalitis", "Infección de garganta",
    "Rinitis", "Sinusitis crónica", "Gastroenteritis", "Mononucleosis infecciosa", "Dengue",
    "Malaria", "Fiebre tifoidea", "Chikungunya", "Meningitis", "Sepsis", "Bronconeumonía",
    "Infección respiratoria viral"
]

X_train, X_test, y_train, y_test = train_test_split(sintomas, diagnosticos, test_size=0.2, random_state=42)
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

if enviado:
    prediccion = modelo.predict([inputs])
    st.markdown(f"<div class='diagnostico-box'><strong>✅ Diagnóstico sugerido:</strong> {prediccion[0]}</div>", unsafe_allow_html=True)

with st.expander("📊 Mostrar precisión del modelo"):
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    st.info(f"Precisión del modelo: {precision * 100:.2f}%")
