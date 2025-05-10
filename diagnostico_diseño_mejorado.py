import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diagnóstico Médico con IA", layout="centered")

st.image("https://cdn-icons-png.flaticon.com/512/1995/1995471.png", width=80)
st.markdown("<h1 style='text-align: center; color: #6c63ff;'>🧠 Diagnóstico Médico Automatizado con IA</h1>", unsafe_allow_html=True)
st.markdown("### Seleccione los síntomas que presenta el paciente (1 = Sí, 0 = No):")

nombres_sintomas = [
    "Tos", "Fiebre", "Dolor de cabeza", "Fatiga", "Dolor muscular",
    "Dolor de garganta", "Congestión nasal", "Dificultad para respirar", "Pérdida de apetito"
]

inputs = []
for sintoma in nombres_sintomas:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**{sintoma}:**")
    with col2:
        valor = st.radio("", [1, 0], horizontal=True, label_visibility="collapsed")
        inputs.append(valor)

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

if st.button("🔍 Predecir diagnóstico"):
    prediccion = modelo.predict([inputs])
    st.success(f"✅ Diagnóstico sugerido: **{prediccion[0]}**")

with st.expander("📊 Mostrar precisión del modelo"):
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    st.info(f"Precisión del modelo: {precision * 100:.2f}%")
