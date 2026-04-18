# 🏎️ F1 Race Intelligence

## 📌 Overview

**F1 Race Intelligence** is an end-to-end machine learning application designed to predict Formula 1 race outcomes using pre-race performance data. The system leverages qualifying and practice session metrics to generate probabilistic predictions for race winners and provide analytical insights into driver and team performance.

The project is built with a modular, scalable architecture that separates user interface, core prediction logic, data pipelines, and model training workflows, making it suitable for both experimentation and real-world deployment scenarios.

---

## 🚀 Key Features

* 🔮 **Race Winner Prediction**
  Predicts win probabilities for all drivers based on qualifying and practice performance.

* 🥇 **Podium Projection**
  Identifies the top 3 most probable race finishers.

* ⚔️ **Head-to-Head Driver Analysis**
  Compares two drivers across pace, consistency, and historical performance metrics.

* 🗺️ **Circuit Performance Insights**
  Visualizes driver success rates and performance trends across circuits.

* 🎛️ **What-If Simulation Engine**
  Allows manual adjustment of grid positions to simulate alternate race scenarios.

* 📊 **Interactive Dashboard**
  Built with Streamlit for real-time analytics and intuitive visualization.

* 📈 **Backtesting Capability** *(optional)*
  Evaluate model predictions against historical race outcomes.

---

## 🧠 System Architecture

The project follows a clean SaaS-style layered architecture:

```bash id="x3m6lg"
app/              → User Interface (Streamlit)
core/             → Prediction & feature logic
data_pipeline/    → Data ingestion & preprocessing
training/         → Model training workflows
models/           → Trained model artifacts
config/           → Configuration & metadata
```

---

## 🛠️ Technology Stack

* **Language:** Python
* **Frontend:** Streamlit
* **Data Source:** FastF1 API
* **Machine Learning:**

  * LightGBM (classification)
  * Random Forest (grid modeling)
* **Data Processing:** pandas, numpy
* **Visualization:** matplotlib, altair
* **Model Serialization:** joblib

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash id="s6srgx"
git clone https://github.com/skyisme33/F1-Race-Intelligence.git
cd F1-Race-Intelligence
```

---

### 2. Create Virtual Environment (Recommended)

```bash id="6o47lx"
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash id="jtrtws"
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash id="5s4c7t"
streamlit run app/app.py
```

---

## 🔍 How It Works

### 1. Data Collection

* Fetches session data using FastF1 (qualifying + practice)
* Extracts lap times, sector data, weather conditions

### 2. Feature Engineering

Transforms raw data into predictive signals such as:

* Qualifying pace ratio
* Practice pace ratio
* Sector performance ratios
* Consistency metrics
* Grid advantage & confidence
* Weather interaction features
* Driver form & team momentum

### 3. Model Prediction

* Uses trained ML models to compute win probabilities
* Outputs ranked predictions for all drivers

### 4. Visualization

* Displays predictions, comparisons, and insights via interactive UI

---

## 🤖 Machine Learning Pipeline

### Data Processing

* `clean_dataset.py` → cleans raw dataset
* `precompute_session_stats.py` → extracts session-level features

### Feature Engineering

* `feature_engineering.py` → constructs model features

### Training

* `train_model.py` → main prediction model (LightGBM)
* `train_grid_model.py` → grid position modeling

### Inference

* `predict_winner.py` → real-time prediction logic

---

## 📂 Models

Stored inside `/models`:

* `f1_model.pkl` → main winner prediction model
* `model_features.pkl` → feature schema
* `grid_model.pkl` → grid prediction model
* `grid_model_features.pkl` → grid feature schema

---

## 📁 Data Management

* Raw and processed datasets are excluded from the repository for performance reasons.
* Session cache files and intermediate outputs are generated dynamically during runtime.

---

## ⚠️ Important Notes

* Internet connection is required for FastF1 data retrieval.
* Ensure models are present in `/models` directory before running.
* Cache and large datasets are intentionally excluded using `.gitignore`.

---

## 🔮 Future Enhancements

* Real-time race prediction updates
* REST API deployment (FastAPI integration)
* Advanced telemetry-based features
* Automated retraining pipeline
* Cloud deployment (AWS / Docker)

---

## 👨‍💻 Author

**Aakash Chauhan (skyisme33)**
GitHub: https://github.com/skyisme33

---

## 📜 License

This project is licensed under the MIT License.
