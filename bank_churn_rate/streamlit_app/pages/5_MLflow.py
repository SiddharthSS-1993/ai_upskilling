import streamlit as st

st.set_page_config(page_title="MLflow", layout="wide")
st.title("🧪  MLFlow")

st.write(
    """
This page is a helper to  run MLflow locally and confirm where your tracking data is. 

### Run MLflow UI (example)
If your tracking uri points to a local folder like `mlruns/`, run from repo root:

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

else run from repo root:
```bash
mlflow ui
```
"""
)