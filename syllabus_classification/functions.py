import os
import streamlit as st
from syllabus_classification.functions import extract

st.set_page_config(
    page_title="DigiComp Syllabus Klassifikator",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 DigiComp Syllabus Kompetenz-Extraktion")
st.markdown(
    "Extrahiert Kompetenzen aus deutschsprachigen Kursbeschreibungen gegen das **DigiComp-Framework** "
    "inklusive automatischer Zitat- und Label-Validierung (powered by Hugging Face)."
)

# --- SIDEBAR: API & MODELL ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    base_url = "[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
    secret_name = "HF_TOKEN"
    key = st.secrets.get(secret_name, os.getenv(secret_name, ""))

    if not key:
        key = st.text_input("Hugging Face Token (`hf_...`)", type="password")

    model = st.selectbox(
        "Hugging Face Modell",
        [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ],
    )
    framework_ref = st.text_input("Framework Repo (mckoh/...)", value="digicomp3-0")

# --- HAUPTBEREICH ---
course_text = st.text_area(
    "Kursbeschreibung (Deutsch) eingeben:",
    height=250,
    placeholder="Füge hier die Modul- oder Lehrveranstaltungsbeschreibung ein...",
)

if st.button("Kompetenzen extrahieren", type="primary"):
    if not key:
        st.error(f"Bitte hinterlege `{secret_name}` in den Streamlit Secrets oder gib den Token links ein.")
    elif not course_text.strip():
        st.warning("Bitte füge zuerst eine Kursbeschreibung ein.")
    else:
        with st.spinner(f"Analysiere Kursbeschreibung mit `{model}`..."):
            try:
                extraction_result = extract(
                    text=course_text,
                    api_key=key,
                    model=model,
                    base_url=base_url,
                    framework_ref=framework_ref,
                )

                competences = extraction_result.competences
                st.subheader(f"Verifizierte Kompetenzen ({len(competences)})")

                if not competences:
                    st.info(
                        "Es konnten keine Kompetenzen extrahiert werden, die sowohl dem "
                        "DigiComp-Katalog entsprechen als auch wörtlich im Text belegt sind."
                    )
                else:
                    rows = [c.model_dump() for c in competences]
                    st.dataframe(rows, use_container_width=True)

                    with st.expander("JSON-Output anzeigen"):
                        st.json(extraction_result.model_dump())

            except Exception as e:
                st.error(f"Fehler bei der Klassifikation: {e}")