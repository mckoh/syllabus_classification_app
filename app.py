import os
import streamlit as st
from syllabus_classification import extract  # Pfad ggf. an dein Paket anpassen

st.set_page_config(
    page_title="DigiComp Syllabus Klassifikator",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 DigiComp Syllabus Kompetenz-Extraktion")
st.markdown(
    "Extrahiert Kompetenzen aus deutschsprachigen Kursbeschreibungen gegen das **DigiComp-Framework** "
    "inklusive automatischer Zitat- und Label-Validierung."
)

# --- SIDEBAR: API & MODELL ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    provider = st.selectbox(
        "API-Provider",
        ["Groq (Llama 3.1)", "xAI (Grok)"],
    )

    if provider == "Groq (Llama 3.1)":
        base_url = "https://api.groq.com/openai/v1"
        model_options = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        secret_name = "GROQ_API_KEY"
    else:
        base_url = "https://api.x.ai/v1"
        model_options = ["grok-3-mini", "grok-2-latest"]
        secret_name = "XAI_API_KEY"

    default_key = st.secrets.get(secret_name, os.getenv(secret_name, ""))
    api_key = st.text_input(
        f"{secret_name}",
        value=default_key,
        type="password",
        help="Kann für das Deployment auch in .streamlit/secrets.toml hinterlegt werden.",
    )
    selected_model = st.selectbox("Modell", model_options)
    framework_ref = st.text_input("Framework Repo (mckoh/...)", value="digicomp3-0")

# --- HAUPTBEREICH ---
course_text = st.text_area(
    "Kursbeschreibung (Deutsch) eingeben:",
    height=250,
    placeholder="Füge hier die Modul- oder Lehrveranstaltungsbeschreibung ein...",
)

if st.button("Kompetenzen extrahieren", type="primary"):
    if not api_key:
        st.error(f"Bitte gib einen gültigen `{secret_name}` in der Seitenleiste ein.")
    elif not course_text.strip():
        st.warning("Bitte füge zuerst eine Kursbeschreibung ein.")
    else:
        with st.spinner(f"Analysiere Kursbeschreibung mit `{selected_model}`..."):
            try:
                extraction_result = extract(
                    text=course_text,
                    api_key=api_key,
                    model=selected_model,
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
                    # Dynamische Aufbereitung aller Felder deines Pydantic-Modells Extraction
                    rows = [c.model_dump() for c in competences]
                    st.dataframe(rows, use_container_width=True)

                    with st.expander("JSON-Output anzeigen"):
                        st.json(extraction_result.model_dump())

            except Exception as e:
                st.error(f"Fehler bei der Klassifikation: {e}")