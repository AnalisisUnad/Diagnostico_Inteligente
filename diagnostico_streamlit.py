import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Datos: síntomas y diagnóstico
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

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(sintomas, diagnosticos, test_size=0.2, random_state=42)
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# Interfaz Streamlit
st.title("🧠 Diagnóstico Médico Automatizado con IA")
st.write("Seleccione los síntomas que presenta el paciente (1 = Sí, 0 = No):")

# Entrada de síntomas
inputs = []
nombres_sintomas = [
    "Tos", "Fiebre", "Dolor de cabeza", "Fatiga", "Dolor muscular",
    "Dolor de garganta", "Congestión nasal", "Dificultad para respirar", "Pérdida de apetito"
]

for sintoma in nombres_sintomas:
    valor = st.radio(f"{sintoma}:", [1, 0], horizontal=True)
    inputs.append(valor)

# Botón para predecir
if st.button("Predecir diagnóstico"):
    prediccion = modelo.predict([inputs])
    st.success(f"✅ Diagnóstico sugerido: **{prediccion[0]}**")

# Mostrar precisión
if st.checkbox("Mostrar precisión del modelo"):
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    st.info(f"Precisión del modelo: {precision * 100:.2f}%")
