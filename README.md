# 🏡 California House Price Predictor

This is a simple and interactive web app built using *Streamlit* and *Scikit-learn* to predict median house prices based on various housing features.

## 🚀 Features

- Predicts house price based on:
  - Location (longitude, latitude)
  - Age, rooms, population, income, and more
  - Ocean proximity (categorical)
- Built with a preprocessing pipeline
- Model: Random Forest Regressor
- Real-time prediction interface with a clean UI

## 🛠 Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit

## 🧠 How to Run

```bash
pip install -r requirements.txt
streamlit run house_price_app.py