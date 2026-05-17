import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

with open("models/predictive_maintenance_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ============================================================
# LOAD DATASET
# ============================================================

columns = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

df.columns = columns

# ============================================================
# FRIENDLY SENSOR NAMES
# ============================================================

sensor_labels = {
    "sensor_2": "Temperature Sensor",
    "sensor_3": "Pressure Sensor",
    "sensor_4": "Vibration Sensor",
    "sensor_7": "Fuel Flow Sensor",
    "sensor_8": "Thermal Sensor",
    "sensor_9": "Rotor Speed Sensor",
    "sensor_11": "Engine Stability Sensor",
    "sensor_12": "Combustion Sensor",
    "sensor_13": "Performance Sensor",
    "sensor_14": "Efficiency Sensor",
    "sensor_15": "Operational Sensor",
    "sensor_17": "Mechanical Stress Sensor",
    "sensor_20": "Health Monitoring Sensor",
    "sensor_21": "Failure Indicator Sensor"
}

# ============================================================
# FEATURE RENAME FUNCTION
# ============================================================

def rename_feature(feature):

    # Rolling mean features
    if "_roll_mean" in feature:

        base = feature.replace("_roll_mean", "")

        return (
            sensor_labels.get(base, base)
            + " (Rolling Avg)"
        )

    # Normal sensor names
    return sensor_labels.get(feature, feature)

# ============================================================
# CREATE RUL
# ============================================================

rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]

df = df.merge(rul, on="unit")

df["RUL"] = df["max_cycle"] - df["cycle"]

df.drop("max_cycle", axis=1, inplace=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

sensors = [col for col in df.columns if "sensor" in col]

for sensor in sensors:

    df[f"{sensor}_roll_mean"] = (
        df.groupby("unit")[sensor]
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

df = df.bfill().ffill()

# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Predictive Maintenance Dashboard")

st.markdown("""
AI-powered predictive maintenance system using sensor-based machine learning.

This dashboard predicts whether an industrial engine is at risk of failure.
""")

st.divider()

# ============================================================
# ENGINE SELECTION
# ============================================================

engine_ids = sorted(df["unit"].unique())

selected_engine = st.sidebar.selectbox(
    "Select Engine ID",
    engine_ids
)

# ============================================================
# ENGINE DATA
# ============================================================

engine_data = df[df["unit"] == selected_engine]

# ============================================================
# CYCLE SELECTION
# ============================================================

available_cycles = engine_data["cycle"].tolist()

selected_cycle = st.sidebar.slider(
    "Select Engine Cycle",
    min_value=int(min(available_cycles)),
    max_value=int(max(available_cycles)),
    value=int(max(available_cycles))
)

# ============================================================
# FILTER SELECTED CYCLE
# ============================================================

selected_data = engine_data[
    engine_data["cycle"] == selected_cycle
]

# ============================================================
# PREPARE FEATURES
# ============================================================

X_input = selected_data[feature_columns]

# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(X_input)[0]

probability = model.predict_proba(X_input)[0][1]

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

st.divider()

st.subheader("🧠 Model Explainability (SHAP)")

# Create explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_input)

# Create SHAP explanation dataframe
shap_df = pd.DataFrame({
    "Feature": [rename_feature(col) for col in X_input.columns],
    "Impact": np.abs(shap_values[0])
})

# Top important features
shap_df = shap_df.sort_values(
    by="Impact",
    ascending=False
).head(10)

# Plot
fig_shap, ax_shap = plt.subplots(figsize=(10, 5))

ax_shap.barh(
    shap_df["Feature"],
    shap_df["Impact"]
)

ax_shap.invert_yaxis()

ax_shap.set_title("Top Features Influencing Prediction")

ax_shap.set_xlabel("SHAP Impact")

st.pyplot(fig_shap)

# STATUS SECTION
# ============================================================

st.subheader(
    f"Engine {selected_engine} — Cycle {selected_cycle}"
)

col1, col2, col3 = st.columns(3)

# =========================
# HEALTH STATUS
# =========================

with col1:

    if probability < 0.30:

        st.success("✅ HEALTHY")

    elif probability < 0.70:

        st.warning("⚠️ WARNING")

    else:

        st.error("🔴 CRITICAL")

# =========================
# FAILURE PROBABILITY
# =========================

with col2:

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

# =========================
# CURRENT RUL
# =========================

with col3:

    current_rul = int(selected_data["RUL"].values[0])

    st.metric(
        "Remaining Useful Life",
        current_rul
    )

st.divider()

# ============================================================
# FAILURE RISK BAR
# ============================================================

st.subheader("📊 Failure Risk")

st.progress(float(probability))

# ============================================================
# INTERPRETATION
# ============================================================

st.subheader("🧠 Model Interpretation")

if probability < 0.30:

    st.info("""
    Machine is operating under healthy conditions.

    No immediate maintenance action required.
    """)

elif probability < 0.70:

    st.warning("""
    Early degradation patterns detected.

    Preventive maintenance is recommended.
    """)

else:

    st.error("""
    High probability of failure detected.

    Immediate inspection is strongly recommended.
    """)

st.divider()
# ============================================================
# SENSOR VISUALIZATION
# ============================================================

st.subheader("📈 Sensor Trend Visualization")

sensor_display_names = {
    rename_feature(sensor): sensor
    for sensor in sensors
}

selected_sensor_name = st.selectbox(
    "Select Sensor",
    list(sensor_display_names.keys())
)

sensor_to_plot = sensor_display_names[selected_sensor_name]

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    engine_data["cycle"],
    engine_data[sensor_to_plot]
)

ax.set_xlabel("Cycle")

ax.set_ylabel(sensor_to_plot)

ax.set_title(
    f"{rename_feature(sensor_to_plot)} Trend"
)

st.pyplot(fig)

# ============================================================
# FAILURE PROBABILITY OVER TIME
# ============================================================

st.divider()

st.subheader("📉 Failure Probability Across Engine Lifecycle")

# Prepare predictions for all cycles

engine_features = engine_data[feature_columns]

all_probabilities = model.predict_proba(engine_features)[:, 1]

# Plot

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(
    engine_data["cycle"],
    all_probabilities
)

ax2.set_xlabel("Cycle")

ax2.set_ylabel("Failure Probability")

ax2.set_title(
    f"Failure Risk Progression - Engine {selected_engine}"
)

st.pyplot(fig2)

# ============================================================
# RAW DATA
# ============================================================

with st.expander("📁 View Engine Sensor Data"):

    st.dataframe(engine_data.tail(20))

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
---
### 🚀 Tech Stack
- Streamlit
- XGBoost
- Pandas
- NASA CMAPSS Dataset

Built for predictive maintenance and industrial AI applications.
""")