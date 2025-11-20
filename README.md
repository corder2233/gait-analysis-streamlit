# 🏃 Human Gait Analysis using Reinforcement Learning (DQN & PPO)

A complete Streamlit-based web application that predicts human gait activities using Reinforcement Learning models (DQN & PPO) trained on wearable sensor data from the HuGaDB dataset. This system provides real-time activity classification, model performance evaluation, and rich interactive visualizations for academic and research demonstration purposes.

---

## 🌐 Live Application

🔗 **Live Demo URL:**
[https://human-gait-analysis.streamlit.app/](https://human-gait-analysis.streamlit.app/)

You can access the deployed application directly using the link above. Upload a sensor CSV file containing an `activity` column to see real-time predictions and analytics.

---

## 🎯 Project Objectives

* Predict human gait activities using Reinforcement Learning techniques
* Implement Deep Q-Network (DQN) and Proximal Policy Optimization (PPO)
* Visualize sensor signals and performance metrics
* Provide a user-friendly dashboard for real-time interaction
* Demonstrate ML concepts in an academic minor project

---

## 🚀 Key Features

* ✅ Activity classification using DQN & PPO
* ✅ CSV file upload for live prediction
* ✅ Real-time accuracy calculation
* ✅ Confusion matrix visualization
* ✅ Sensor signal plots (Accelerometer, Gyroscope, EMG)
* ✅ Precision, Recall, F1-Score & MSE graphs
* ✅ Intuitive and responsive Streamlit UI

---

## 🧠 Technologies Used

| Category      | Tools & Libraries            |
| ------------- | ---------------------------- |
| Language      | Python                       |
| Frontend      | Streamlit                    |
| ML Models     | Stable-Baselines3 (DQN, PPO) |
| Data Handling | Pandas, NumPy                |
| Visualization | Plotly, Matplotlib           |
| Evaluation    | Scikit-learn                 |
| Deployment    | Streamlit Community Cloud    |

---

## 📁 Project Structure

```
gait-analysis-streamlit/
│
├── app.py                 # Main Streamlit application
├── visualization.py       # Graph and plotting functions
├── requirements.txt       # Required Python dependencies
├── models/
│   ├── HuGaDB_Model_DQN.zip
│   └── HuGaDB_Model_PPO.zip
└── README.md
```

---

## 📊 How It Works

1. User selects the RL model (DQN or PPO)
2. Uploads a CSV file containing sensor data
3. Application processes features and performs predictions
4. Results are displayed with:

   * Accuracy score
   * Confusion matrix
   * Activity comparison table
   * Sensor signal graphs
   * Performance trend graphs

---

## ⚙️ Installation & Setup (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/corder2233/gait-analysis-streamlit.git
cd gait-analysis-streamlit
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open browser and visit:

```
http://localhost:8501
```

---

## 📥 CSV File Requirements

The uploaded CSV file must contain:

* An `activity` column
* Sensor features such as:

  * accelerometer_x
  * gyroscope_y
  * EMG_signal

Example structure:

```
acc_x, acc_y, gyro_x, EMG1, activity
0.12, 0.45, 0.67, 0.22, walking
```

---

## 📈 Output Insights

The app provides:

* Predicted activity labels
* Accuracy percentage
* Confusion Matrix heatmap
* Colored comparison table (actual vs predicted)
* Precision, Recall, F1 & MSE trends over time

---

## 🔒 License and Usage

This project is intended for **academic and demonstration purposes only**. The source code is protected and not intended for reuse or redistribution without explicit permission from the author.

---

## 👨‍💻 Author

**(Corder2233)**
Minor Project – Human Gait Analysis using Reinforcement Learning
Department of Engineering

GitHub: [https://github.com/corder2233](https://github.com/corder2233)

---

## ✅ Future Improvements

* Real-time sensor data integration
* Mobile-friendly UI version
* Support for more activity classes
* Advanced model comparison dashboard

---

## ⭐ Acknowledgements

* HuGaDB Dataset
* Stable-Baselines3 Community
* Streamlit Team

---

If you find this project useful or interesting, feel free to ⭐ the repository!
