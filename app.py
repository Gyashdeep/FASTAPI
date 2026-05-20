import streamlit as str_ln
import requests
import pandas as pd

# -------------------------------------------------------------
# 1. UI CONFIGURATION & TERMINAL THEME OVERRIDES
# -------------------------------------------------------------
str_ln.set_page_config(
    page_title="Inference Node",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for dark, responsive UI elements
str_ln.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div.stButton > button:first-child {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_index=True)

# API Configuration
API_URL = "http://127.0.0.1:8000/predict"

# -------------------------------------------------------------
# 2. SIDEBAR - FEATURE CHANNELS
# -------------------------------------------------------------
str_ln.sidebar.title("⚡ Input Channels")
str_ln.sidebar.markdown("Adjust raw dimensions for real-time calculation.")

sepal_length = str_ln.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, step=0.1)
sepal_width  = str_ln.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.5, step=0.1)
petal_length = str_ln.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, step=0.1)
petal_width  = str_ln.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 0.2, step=0.1)

# Assemble request payload
payload = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}

# -------------------------------------------------------------
# 3. MAIN COCKPIT - EXECUTION & MONITORING
# -------------------------------------------------------------
str_ln.title("⚡ INFERENCE WORKSPACE // SYSTEM ACTIVE")
str_ln.write("Streamlit interface hooked directly into the high-performance FastAPI back-end.")

col1, col2 = str_ln.columns([1, 1.5])

with col1:
    str_ln.subheader("📊 Target State Vectors")
    # Quick tabular summary of inputs
    input_df = pd.DataFrame([payload])
    str_ln.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
    
    trigger_execution = str_ln.button("RUN INFERENCE SEQUENCE")

with col2:
    str_ln.subheader("🧠 Model Response Matrix")
    
    if trigger_execution:
        with str_ln.spinner("Polling FastAPI Inference Engine..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Core Class Result Display
                    str_ln.success(f"**PREDICTED CLASS:** {data['class_label'].upper()}")
                    str_ln.metric(label="Target Class Index", value=data['class_index'])
                    
                    # Probabilities Processing
                    classes = ["Setosa", "Versicolor", "Virginica"]
                    prob_df = pd.DataFrame({
                        "Target Class": classes,
                        "Confidence Factor": data["probabilities"]
                    })
                    
                    str_ln.write("##### Probability Profile:")
                    str_ln.bar_chart(prob_df.set_index("Target Class"))
                    
                else:
                    str_ln.error(f"Engine Exception: HTTP Status {response.status_code}")
                    str_ln.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                str_ln.error("CRITICAL: Connection Refused. Ensure your FastAPI server is running at http://127.0.0.1:8000")
            except Exception as e:
                str_ln.error(f"Runtime Failure: {str(e)}")
    else:
        str_ln.info("System idling. Adjust features in the control panel and click 'RUN INFERENCE SEQUENCE'.")
