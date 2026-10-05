import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Transcriptomics KEGG Pipeline",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Transcriptomics & KEGG Pathway Analyzer")

st.write(
    "A pipeline for differential expression analysis, "
    "gene annotation, and KEGG pathway mapping."
)

st.header("Expression Data")

data = pd.read_csv("dummy_data.csv")

st.dataframe(data, use_container_width=True)
