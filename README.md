# 🐧 Penguin Species Prediction App

A **Machine Learning web application built with Streamlit** that predicts the **species of Palmer Penguins** based on their physical measurements.

The model is trained using a **Random Forest Classifier** and allows users to input penguin features manually or upload a CSV file to make predictions.

---

# 🚀 Features

- Predict penguin species using machine learning  
- Interactive **Streamlit web interface**  
- Manual input using sliders and dropdown menus  
- Upload CSV file for prediction  
- Displays **prediction probabilities**  
- Uses a trained **Random Forest model**

---

# 📊 Features Used for Prediction

The model predicts the penguin species using the following attributes:

- Island  
- Bill Length (mm)  
- Bill Depth (mm)  
- Flipper Length (mm)  
- Body Mass (g)  
- Sex  

These features are processed and encoded before being passed to the trained model.

---

# 🧠 Machine Learning Model

Algorithm used:

**Random Forest Classifier**

The model is trained on the **Palmer Penguins dataset** and saved as a serialized `.pkl` file.

Training and preprocessing steps include:

- Data loading
- Encoding categorical features
- Feature selection
- Model training
- Saving the trained model using `pickle`

---
