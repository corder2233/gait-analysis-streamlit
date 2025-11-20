import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
)
from stable_baselines3 import DQN, PPO

# -------------------- Paths (relative, for deployment) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# -------------------- Model Loading --------------------
@st.cache_resource
def load_dqn(model_path, env=None):
    return DQN.load(model_path, env=env)


@st.cache_resource
def load_ppo(model_path, env=None):
    return PPO.load(model_path, env=env)


def predict_action(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    # Make sure we return a scalar (for metrics)
    if isinstance(action, np.ndarray):
        return int(action.squeeze())
    return int(action)


# -------------------- Plotting Functions --------------------
def plot_sensor_signals(df, sensors):
    fig = go.Figure()
    for sensor in sensors:
        if sensor in df.columns:
            fig.add_trace(go.Scatter(y=df[sensor], mode="lines", name=sensor))
    fig.update_layout(
        title="Sensor Signals",
        xaxis_title="Sample Index",
        yaxis_title="Sensor Value",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        title="Confusion Matrix",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_side="top",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_actual_vs_predicted_table(y_true, y_pred, activity_names):
    actual_labels = [activity_names[i] for i in y_true]
    predicted_labels = [activity_names[i] for i in y_pred]
    df_compare = pd.DataFrame(
        {"Actual Activity": actual_labels, "Predicted Activity": predicted_labels}
    )

    def color_row(row):
        return [
            "color: green"
            if row["Actual Activity"] == row["Predicted Activity"]
            else "color: red"
            for _ in row
        ]

    return df_compare.style.apply(color_row, axis=1)


# -------------------- Streamlit Page Config --------------------
st.set_page_config(
    page_title="Human GAIT Prediction", layout="wide", page_icon="🤖"
)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_type = st.selectbox("Select RL Model", ["DQN", "PPO"])

    # Use relative paths so it works on Streamlit Cloud
    default_model_path = (
        os.path.join(MODELS_DIR, "HuGaDB_Model_DQN.zip")
        if model_type == "DQN"
        else os.path.join(MODELS_DIR, "HuGaDB_Model_PPO.zip")
    )

    st.markdown("### Model file path (inside this app)")
    model_path = st.text_input("Model Path", value=default_model_path)

    uploaded_file = st.file_uploader("Upload Sensor CSV", type=["csv"])

# -------------------- Main Content --------------------
st.title("🏃 Human GAIT Prediction")
st.markdown(
    """
This app performs activity classification using RL models **DQN** and **PPO**
trained on the HuGaDB dataset.  

Upload your sensor CSV (with an **'activity'** column) and select the model to get started.
"""
)

if uploaded_file and os.path.isfile(model_path):
    # Read data
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Sample Data")
    st.dataframe(df.head())

    if "activity" not in df.columns:
        st.error("❌ The uploaded CSV must contain an 'activity' column.")
        st.stop()

    # Prepare data
    feature_cols = [c for c in df.columns if c != "activity"]
    X = df[feature_cols].values.astype(np.float32)
    y_true_cat = df["activity"].astype("category")
    y_true = y_true_cat.cat.codes.values
    activity_names = y_true_cat.cat.categories.tolist()

    st.success(
        f"✅ Loaded {len(X)} samples with {len(activity_names)} activity classes."
    )

    # Load model
    with st.spinner(f"Loading {model_type} model..."):
        try:
            if model_type == "DQN":
                model = load_dqn(model_path)
            else:
                model = load_ppo(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    # Predict once for all samples
    with st.spinner("Predicting activities..."):
        y_pred = np.array([predict_action(model, obs) for obs in X])

    # Classification accuracy
    acc = accuracy_score(y_true, y_pred) * 100

    # Layout
    col1, col2 = st.columns([2.5, 1.8])

    with col1:
        st.markdown(f"### 🎯 Prediction Accuracy: **{acc:.2f}%**")

        # Sensor plots
        accel = [c for c in df.columns if "accelerometer" in c]
        gyro = [c for c in df.columns if "gyroscope" in c]
        emg = [c for c in df.columns if "EMG" in c]

        if accel:
            with st.expander("Accelerometer Data", expanded=True):
                plot_sensor_signals(df, accel)
        if gyro:
            with st.expander("Gyroscope Data"):
                plot_sensor_signals(df, gyro)
        if emg:
            with st.expander("EMG Signals"):
                plot_sensor_signals(df, emg)

    with col2:
        plot_confusion_matrix(y_true, y_pred, activity_names)
        styled = show_actual_vs_predicted_table(
            y_true, y_pred, activity_names
        )
        st.subheader("Actual vs Predicted Activities")
        st.dataframe(styled)

    # -------------------- Metric Curves Over Time --------------------
    precision_ph = st.empty()
    recall_ph = st.empty()
    f1_ph = st.empty()
    mse_ph = st.empty()

    precision_vals, recall_vals, f1_vals, mse_vals = [], [], [], []
    sample_counts = []

    # Dummy regression targets
    y_true_reg = np.random.rand(len(y_true))
    y_pred_reg = []

    for i in range(len(X)):
        # dummy regression prediction
        y_pred_reg.append(y_true_reg[i] + np.random.normal(0, 0.1))

        sample_counts.append(i + 1)

        # use already computed y_pred
        y_true_subset = y_true[: i + 1]
        y_pred_subset = y_pred[: i + 1]
        y_true_reg_subset = y_true_reg[: i + 1]
        y_pred_reg_subset = y_pred_reg[: i + 1]

        precision_vals.append(
            precision_score(
                y_true_subset,
                y_pred_subset,
                average="weighted",
                zero_division=0,
            )
        )
        recall_vals.append(
            recall_score(
                y_true_subset,
                y_pred_subset,
                average="weighted",
                zero_division=0,
            )
        )
        f1_vals.append(
            f1_score(
                y_true_subset,
                y_pred_subset,
                average="weighted",
                zero_division=0,
            )
        )
        mse_vals.append(
            mean_squared_error(y_true_reg_subset, y_pred_reg_subset)
        )

        # Update plots every 10 samples or at the end
        if (i + 1) % 10 == 0 or (i + 1) == len(X):
            # Precision
            fig_p = go.Figure()
            fig_p.add_trace(
                go.Scatter(
                    x=sample_counts,
                    y=precision_vals,
                    mode="lines+markers",
                    name="Precision",
                )
            )
            fig_p.update_layout(
                title="📈 Precision Over Time",
                xaxis_title="Sample Count",
                yaxis_title="Precision",
                template="plotly_white",
                margin=dict(l=40, r=10, t=40, b=30),
            )
            precision_ph.plotly_chart(fig_p, use_container_width=True)

            # Recall
            fig_r = go.Figure()
            fig_r.add_trace(
                go.Scatter(
                    x=sample_counts,
                    y=recall_vals,
                    mode="lines+markers",
                    name="Recall",
                )
            )
            fig_r.update_layout(
                title="📈 Recall Over Time",
                xaxis_title="Sample Count",
                yaxis_title="Recall",
                template="plotly_white",
                margin=dict(l=40, r=10, t=40, b=30),
            )
            recall_ph.plotly_chart(fig_r, use_container_width=True)

            # F1
            fig_f1 = go.Figure()
            fig_f1.add_trace(
                go.Scatter(
                    x=sample_counts,
                    y=f1_vals,
                    mode="lines+markers",
                    name="F1-Score",
                )
            )
            fig_f1.update_layout(
                title="📈 F1-Score Over Time",
                xaxis_title="Sample Count",
                yaxis_title="F1-Score",
                template="plotly_white",
                margin=dict(l=40, r=10, t=40, b=30),
            )
            f1_ph.plotly_chart(fig_f1, use_container_width=True)

            # MSE
            fig_mse = go.Figure()
            fig_mse.add_trace(
                go.Scatter(
                    x=sample_counts,
                    y=mse_vals,
                    mode="lines+markers",
                    name="MSE",
                )
            )
            fig_mse.update_layout(
                title="📉 Mean Squared Error (MSE) Over Time",
                xaxis_title="Sample Count",
                yaxis_title="MSE",
                template="plotly_white",
                margin=dict(l=40, r=10, t=40, b=30),
            )
            mse_ph.plotly_chart(fig_mse, use_container_width=True)

        time.sleep(0.05)

else:
    if not uploaded_file:
        st.info("Please upload a CSV file to start.")
    if not os.path.isfile(model_path):
        st.warning("⚠️ Please enter a valid model file path in the sidebar.")
