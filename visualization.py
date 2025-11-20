import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load the Dataset ---
csv_path = r"C:\Users\jenas\OneDrive\Documents\Projects\gait_prediction\gait_prediction\model v-1\app_streamlit\HuGaDB_v2_various_01_00.csv"
df = pd.read_csv(csv_path)

# --- Clean column names (remove extra spaces) ---
df.columns = df.columns.str.strip()

# --- Plot Multiple Time Series (x, y, z) ---
def plot_time_series(df, cols, title):
    plt.figure(figsize=(12, 4))
    for c in cols:
        if c in df.columns:
            plt.plot(df[c], label=c)
        else:
            print(f"❌ Column not found: {c}")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sensor Value")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# --- Plot Magnitude from x, y, z components ---
def plot_magnitude(df, prefix, title):
    x_col, y_col, z_col = f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"
    if all(col in df.columns for col in [x_col, y_col, z_col]):
        mag = np.linalg.norm(df[[x_col, y_col, z_col]].values, axis=1)
        plt.figure(figsize=(12, 3))
        plt.plot(mag, label=f"{prefix} Magnitude", color='darkorange')
        plt.title(title)
        plt.ylabel("Magnitude")
        plt.xlabel("Sample Index")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ Missing columns for {prefix}: {x_col}, {y_col}, {z_col}")

# --- Parts to visualize ---
body_parts = ["right_thigh", "right_shin", "right_foot"]

# --- Loop through each part for plotting ---
for part in body_parts:
    # Plot Accelerometer Axes
    acc_cols = [f"accelerometer_{part}_x", f"accelerometer_{part}_y", f"accelerometer_{part}_z"]
    plot_time_series(df, acc_cols, f"Accelerometer — {part.replace('_', ' ').title()}")

    # Plot Gyroscope Axes
    gyro_cols = [f"gyroscope_{part}_x", f"gyroscope_{part}_y", f"gyroscope_{part}_z"]
    plot_time_series(df, gyro_cols, f"Gyroscope — {part.replace('_', ' ').title()}")

    # Plot Accelerometer Magnitude
    plot_magnitude(df, f"accelerometer_{part}", f"Accel Magnitude — {part.replace('_', ' ').title()}")
