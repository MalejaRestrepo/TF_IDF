import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ------------------ CONFIGURACIÓN GENERAL ------------------
st.set_page_config(
    page_title="Análisis TF-IDF",
    page_icon="🔍",
    layout="wide"
)

# ------------------ ESTILO GLOBAL ------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #e6e4ff 0%, #d9f4ff 100%);
    color: #1f244b;
    font-family: 'Poppins', sans-serif;
}
.block-container {
    background: #f9faff;
    border-radius: 16px;
    border: 1px solid #c0d3ff;
    padding: 2rem 2.2rem;
    box-shadow: 0 8px 24px rgba(31, 36, 75, 0.12);
}
h1, h2, h3 {
    color: #1f244b;
    font-weight: 700;
    text-align: center;
}
label, p, div, span {
    color: #1f244b !important;
}
section[data-testid="stSidebar"] {
    background: #eaf3ff;
    border-right: 2px solid #bcd6ff;
}
section[data-testid="stSidebar"] * {
    color: #1e1c3a !important;
}
div.stButton > button {
    background: linear-gradient(90deg, #b9a6ff 0%, #9be4ff 100%) !important;
    color: #1f244b !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: 1px solid #9fcaff !important;
    box-shadow: 0 6px 14px rgba(31, 36, 75, 0.18) !important;
    font-size: 16px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #a694ff 0%, #8fd8ff 100%) !important;
    transform: translateY(-1px);
}
textarea, .stTextInput input {
    background-color: #ffffff !important;
    color: #1f244b !important;
    border-radius: 10px !important;
    border: 1px solid #bda5ff !important;
}
table, .stDataFrame {
    border-radius: 10px !important;
    border: 1px solid #cbd9ff !important;
    background-color: white !important;
}
[data-testid="stHeader"] {
    background: linear-gradient(90deg, #7c9eff 0%, #b0c3ff 100%) !important;
    color: white !important;
    height: 3.5rem;
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.25);
}
</style>
""", unsafe_allow_html=True)

# ------------------ CONTENIDO PRINCIPAL ------------------
st.title("🔍 Demo de TF-IDF con Preguntas y Respuestas")
st.markdown("""
Cada línea se trata como un **documento** (puede ser una frase, un párrafo o un texto más largo).  
⚠️ Los documentos y las preguntas deben estar en **inglés**, ya que el análisis está configurado para ese idioma.  
La aplicación aplica normalización y *stemming* para que palabras como *playing* y *play* se consideren equivalentes.
""")

# ------------------ ENTRADA DE TEXTO ------------------
text_input = st.text_area(
    "📘 Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)
question = st.text_input("❓ Escribe una pregunta (en inglés):", "Who is playing?")

# ------------------ FUNCIONES DE PROCESAMIENTO ------------------
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]

# ------------------ LÓGICA PRINCIPAL ------------------
if st.button("Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.markdown("### 🧮 Matriz TF-IDF (stems)")
        st.dataframe(df_tfidf.round(3))

        # Calcular similitud coseno con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()

        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 💬 Resultado del análisis")
        st.write(f"**Tu pregunta:** {question}")
        st.success(f"**Documento más relevante (Doc {best_idx+1}):** {best_doc}")
        st.write(f"**Puntaje de similitud:** `{best_score:.3f}`")

        # Tabla con todas las similitudes
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        }).sort_values("Similitud", ascending=False)

        st.markdown("### 📊 Puntajes de similitud")
        st.dataframe(sim_df.style.format({"Similitud": "{:.3f}"}))

        # Mostrar coincidencias de stems
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

        st.markdown("### 🧩 Stems de la pregunta presentes en el documento elegido")
        if matched:
            st.info(", ".join(matched))
        else:
            st.warning("No se encontraron coincidencias exactas de stems.")

# ------------------ PIE DE PÁGINA ------------------
st.markdown("---")
st.markdown("Desarrollado con 💜 usando Streamlit, Scikit-learn y NLTK")

