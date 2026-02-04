 #🌾 AgriAI – Farmer Yield Prediction & Advisory System

AgriAI is a machine learning–based agricultural decision support system that predicts crop yield and provides actionable farming recommendations using climate data, pesticide usage, and crop growth stage.  
The project is designed to be simple, farmer-friendly, and practical for real-world use.

---

 ## 📌 Problem Statement
Farmers often make critical decisions on irrigation, fertilizer, and pest control without data-driven insights. This can lead to low yield, excess cost, and inefficient resource usage.  
AgriAI addresses this problem by using historical agricultural and weather data to predict crop yield and recommend optimal actions.

---

## 🎯 Objectives
- Predict crop yield using machine learning techniques  
- Provide real-time farming recommendations  
- Use weather data to adjust irrigation decisions  
- Offer a simple and interactive user interface for farmers  
- Support multiple regional languages for accessibility  

---

## 🚀 Features
- Crop yield prediction (hg/ha)
- Confidence interval using quantile regression (P10–P90)
- Weather-aware irrigation recommendations
- Fertilizer and pest control suggestions
- Multilingual support (English, Hindi, Telugu, Tamil, Kannada)
- Interactive Streamlit dashboard
- What-if analysis for rainfall and temperature
- Built-in farmer assistant chatbot

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib  
- **Web App Framework:** Streamlit  
- **Model Persistence:** Joblib  
- **APIs:** Open-Meteo, NASA POWER (weather data)

---
##📂 Project Structure
AgriAI-Farmer-Advisory/
├── app.py                # Streamlit UI
├── model_training.py     # Model training & saving
├── chatbot.py            # Farmer assistant
├── recommendations.py   # Advisory logic
├── utils.py              # Data & weather utilities
├── checks.py             # Data validation
├── data/                 # Datasets
├── models/               # Trained models
├── requirements.txt
├── README.md


## How to Run:
pip install -r requirements.txt
python model_training.py
streamlit run app.py

