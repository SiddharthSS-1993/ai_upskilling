import streamlit as st
import os
import sys 
main=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main)
from streamlit_app.ui_utils import ROOT, list_images
from src.utils.file_utils import load_json

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
st.title("📊 Exploratory Data Analysis")

EDA_DIRECTORY = os.path.join(ROOT, "reports", "eda")

st.write(f"Showing plots from {EDA_DIRECTORY}")

images = list_images(EDA_DIRECTORY)
if not images:
    st.warning(f"No EDA Images found. Make sure you save EDA images to {EDA_DIRECTORY}")
else:
    # grid view
    columns = st.columns(2)
    for i, image in enumerate(images):
        with columns[i % 2]:
            st.image(str(image), caption=image.name, use_container_width=True)
        
    st.subheader("EDA Summary (JSON)")
    SUMMARY_PATH = os.path.join(EDA_DIRECTORY, "metadata", "eda_summary.json")
    summary = load_json(SUMMARY_PATH)

    if summary is None:
        st.info(f"No `eda_summary.json` found in {SUMMARY_PATH[:, -len("eda_summary.json")]}")
    else:
        st.json(summary)

