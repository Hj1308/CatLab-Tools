import io
import os

import pandas as pd
import streamlit as st
from openai import OpenAI


# -------------------------------------------------------------------
# Groq / OpenAI-compatible client
# -------------------------------------------------------------------
MODEL_NAME = "llama-3.3-70b-versatile"


def get_groq_client() -> OpenAI:
    """Create a Groq-compatible OpenAI client from st.secrets or env."""
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to .streamlit/secrets.toml as:\n\n"
            'GROQ_API_KEY = "sk-xxxxxxxxxxxxxxxx"\n'
        )
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


def ask_scientific_question(question: str, context: str = "") -> str:
    """Ask Groq LLM an ODS/catalysis domain question."""
    client = get_groq_client()
    prompt = f"""You are an expert in Oxidative Desulfurization and environmental catalysis.
Use the following context if relevant: {context}

Question: {question}

Give a clear, scientific, and concise answer.
Use formal academic English, structured in short paragraphs.
Where possible, include 2-3 recent peer-reviewed references
with journal name and year (no URLs)."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
        max_tokens=800,
    )
    return response.choices[0].message.content


# -------------------------------------------------------------------
# Sidebar configuration
# -------------------------------------------------------------------
def build_ods_config() -> dict:
    st.sidebar.header("ODS experiment settings")
    substrate_name = st.sidebar.text_input("Substrate (pollutant)", "DBT")
    catalyst_name = st.sidebar.text_input("Catalyst name", "MoS2-based catalyst")
    c0_val = st.sidebar.number_input("Initial S concentration (c0)", 500.0)
    c0_unit = st.sidebar.selectbox("c0 unit", ["ppmS", "mg/L"])
    mw_poll = st.sidebar.number_input("Pollutant molecular weight (g/mol)", 184.26)
    n_sulfur = st.sidebar.number_input("Number of S atoms per molecule", 1, step=1)
    solvent_name = st.sidebar.text_input("Fuel / solvent", "n-Heptane")
    rho = st.sidebar.number_input("Fuel density (g/mL)", 0.684)
    V_fuel = st.sidebar.number_input("Fuel volume (L)", 0.01, step=0.001)
    m_cat = st.sidebar.number_input("Catalyst mass (g)", 0.01, step=0.001)
    temp_C = st.sidebar.number_input("Reaction temperature (\u00b0C)", 60.0)
    O_S = st.sidebar.number_input("O/S molar ratio", 4.0)
    return {
        "substrate_name": substrate_name,
        "catalyst_name": catalyst_name,
        "c0_val": c0_val,
        "c0_unit": c0_unit,
        "mw_poll": mw_poll,
        "n_sulfur": n_sulfur,
        "solvent_name": solvent_name,
        "rho": rho,
        "V_fuel": V_fuel,
        "m_cat": m_cat,
        "temp_C": temp_C,
        "O_S": O_S,
    }


# -------------------------------------------------------------------
# Advanced Excel template
# -------------------------------------------------------------------
def create_advanced_template(cfg, filename="ODS_Advanced_Template_With_Metadata.xlsx"):
    """Create advanced Excel template with Metadata sheet from sidebar settings."""
    metadata = {
        "Parameter": [
            "sample_name", "catalyst_name", "substrate", "c0_value", "c0_unit",
            "mw_pollutant", "n_sulfur", "fuel_solvent", "rho_g_per_mL",
            "V_fuel_mL", "m_cat_mg", "temperature_C", "O_S_ratio",
            "active_sites_mmol_g", "notes",
        ],
        "Value": [
            "DBT-Test-01",
            cfg.get("catalyst_name", "MoS2"),
            cfg.get("substrate_name", "DBT"),
            cfg.get("c0_val", 500),
            cfg.get("c0_unit", "ppmS"),
            cfg.get("mw_poll", 184.26),
            cfg.get("n_sulfur", 1),
            cfg.get("solvent_name", "n-Heptane"),
            cfg.get("rho", 0.684),
            round(cfg.get("V_fuel", 0.01) * 1000, 1),
            round(cfg.get("m_cat", 0.01) * 1000, 1),
            cfg.get("temp_C", 60),
            cfg.get("O_S", 4.0),
            0.5,
            "My experimental ODS data",
        ],
        "Unit/Description": [
            "-", "-", "-", "-", "-", "g/mol", "-", "-", "g/mL",
            "mL", "mg", "\u00b0C", "-", "mmol/g", "-",
        ],
    }
    df_meta = pd.DataFrame(metadata)
    df_raw = pd.DataFrame({
        "Time (min)": [0, 15, 30, 45, 60, 90, 120, 180, 240],
        "Cat-A Removal (%)": [0, 18, 35, 52, 68, 82, 91, 96, 98],
        "Cat-B Removal (%)": [0, 22, 41, 59, 74, 87, 94, 97, 99],
        "Notes": ["" for _ in range(9)],
    })
    instructions = pd.DataFrame({
        "Instructions": [
            "1. Review and edit the Metadata sheet if needed.",
            "2. Fill in Time (min) and Removal (%) columns in Raw_Data sheet.",
            "3. Save the file and upload it in the app.",
            "4. The app will automatically convert units, fit all models "
            "(including second-order), and select the best one using AIC.",
        ]
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_meta.to_excel(writer, sheet_name="Metadata", index=False)
        df_raw.to_excel(writer, sheet_name="Raw_Data", index=False)
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_len:
                            max_len = len(str(cell.value))
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 2, 40)
    buf.seek(0)
    return buf, filename


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ODS kinetic helper", layout="wide")
    st.title("Oxidative Desulfurization \u2013 Kinetic helper")

    cfg = build_ods_config()

    # ---------------------------------------------------------------
    # Template section
    # ---------------------------------------------------------------
    with st.expander("\U0001f4cb Template & upload instructions", expanded=False):
        st.markdown("""
**Required columns:**
- `Time (min)` \u2014 reaction time
- One or more catalyst columns with `Removal (%)` values (0\u2013100)
        """)
        col1, col2 = st.columns(2)
        with col1:
            tmpl = pd.DataFrame({
                "Time (min)": [0, 30, 60],
                "Removal (%)": [0, 50, 80],
            })
            buf_simple = io.BytesIO()
            tmpl.to_excel(buf_simple, index=False)
            buf_simple.seek(0)
            st.download_button(
                "\u2b07\ufe0f Download simple template",
                buf_simple.getvalue(),
                "ods_template_simple.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="simple_template",
            )
        with col2:
            if st.button(
                "\U0001f4cb Generate Advanced Template (with Metadata)",
                type="primary",
                key="adv_template_btn",
            ):
                buf_adv, fname = create_advanced_template(cfg)
                st.download_button(
                    "\u2b07\ufe0f Download Advanced Template",
                    buf_adv.getvalue(),
                    fname,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="adv_template_dl",
                )

    st.markdown("---")

    # ---------------------------------------------------------------
    # AI ODS assistant
    # ---------------------------------------------------------------
    st.subheader("\U0001f52c AI assistant for ODS and environmental catalysis")
    st.markdown(
        "Ask domain-specific questions about oxidative desulfurization, "
        "kinetic modelling, or catalyst performance. "
        "The assistant uses a Groq\u2011hosted LLM (LLaMA\u20113.x) in the background."
    )

    user_q = st.text_area(
        "Your question:",
        key="ods_ai_question",
        height=140,
        placeholder=(
            "Example: How does increasing O/S ratio affect the apparent "
            "rate constant in biphasic ODS systems?"
        ),
    )

    context_lines = [
        f"Substrate: {cfg.get('substrate_name')} ({cfg.get('c0_val')} {cfg.get('c0_unit')})",
        f"Catalyst: {cfg.get('catalyst_name')}",
        f"Fuel: {cfg.get('solvent_name')} (rho={cfg.get('rho')} g/mL, V={cfg.get('V_fuel')} L)",
        f"m_cat={cfg.get('m_cat')} g, T={cfg.get('temp_C')} \u00b0C, O/S={cfg.get('O_S')}",
    ]
    extra_context = "\n".join(context_lines)

    if st.button("Ask AI", type="primary", key="ask_ai_btn"):
        if not user_q.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Contacting AI backend..."):
                try:
                    answer = ask_scientific_question(user_q, context=extra_context)
                    st.markdown("**Answer:**")
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"Error while contacting AI backend: {exc}")


if __name__ == "__main__":
    main()
