import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Load the Large Kaggle Dataset
# Ensure the file name matches exactly what you downloaded
df = pd.read_csv('USA_Housing.csv')

# 2. Prepare the Data
# We will use 'Avg. Area Income', 'Avg. Area House Age', and 'Avg. Area Number of Rooms' 
# to predict the 'Price'
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms']]
y = df['Price']

# 3. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Streamlit UI
st.title("🏠 Advanced House Price Predictor")
st.write("This model uses a Kaggle dataset of 5,000 homes to predict prices.")

# User Inputs
income = st.number_input("Average Area Income:", value=60000)
age = st.slider("Average House Age:", 1, 10, 5)
rooms = st.slider("Average Number of Rooms:", 1, 10, 6)

if st.button("Predict Price"):
    # Create a features array from user input
    user_input = np.array([[income, age, rooms]])
    prediction = model.predict(user_input)
    
    st.success(f"The estimated market value is ${prediction[0]:,.2f}")
    
    # Show accuracy metric for the interview
    st.info("Model Info: This is a Linear Regression model trained on 4,000 samples.")