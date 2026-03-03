import streamlit as st
import os
import sys
from pathlib import Path
main = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main)

from streamlit_app.ui_utils import ROOT, list_images, available_models_in_directory

sampling_methods = ["SMOTE", "ADASYN"]

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("🧠 Explainability (SHAP)")

method = st.selectbox("Sampling Method", sampling_methods, index=0)

# model folders
EXPLAINABILITY_DIRECTORY = Path(os.path.join(ROOT, "reports", "explainability", method))
if not EXPLAINABILITY_DIRECTORY.exists():
    st.warning(f"No explainability folder found at {EXPLAINABILITY_DIRECTORY}")
    st.stop()

model_folders = sorted([p for p in EXPLAINABILITY_DIRECTORY.iterdir() if p.is_dir()])
model_names = [p.name for p in model_folders]

if not model_names:
    st.warning(f"No model folders found in side {EXPLAINABILITY_DIRECTORY}")
    st.stop()

model_name = st.selectbox("Model Name", model_names, index=0)

directory = os.path.join(EXPLAINABILITY_DIRECTORY, model_name)
st.write(f"Showing SHAP plots from `{directory}`")

images = list_images(directory)
if not images:
    st.warning("No SHAP Images found. Ensure you saved youe SHAP Images here.")
else:
    columns = st.columns(2)
    for i, image in enumerate(images):
        with columns[i % 2]:
            st.image(str(image), caption=image.name, use_container_width=True)
