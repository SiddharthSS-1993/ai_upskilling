import streamlit as st
import os
import sys 
main=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main)
from streamlit_app.ui_utils import list_images, ROOT

sampling_method = ["SMOTE", "ADASYN"]

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("📈 Evaluation")
method = st.selectbox("Sampling Method", sampling_method, index=0)

EVALUATION_DIRECTORY = os.path.join(ROOT, "reports", "evaluate", method)

st.write(f"Showing model evaluation plots from `{EVALUATION_DIRECTORY}`")

images = list_images(EVALUATION_DIRECTORY)
if not images:
    st.warning(f"No Evaluation Images found for {method}. Expected under {EVALUATION_DIRECTORY}")
else:
    columns = st.columns(2)
    for i, image in enumerate(images):
        with columns[i % 2]:
            st.image(str(image), caption=image.name, use_container_width=True)


