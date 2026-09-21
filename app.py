
import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('delivery_delay.sav')

# Define the expected feature names based on your training data
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency', 
                 'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction')
st.write('Enter the feature values below to predict delivery delay.')

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    # Using st.number_input for numerical features
    input_data[feature] = st.number_input(f'{feature.replace("_", " ")}:', value=0.0)

if st.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]
    
    st.subheader('Prediction Results:')
    if prediction == 1:
        st.error('Predicted: **Delivery Delayed**')
    else:
        st.success('Predicted: **No Delay**')
    
    st.write(f'Probability of No Delay: {prediction_proba[0]:.4f}')
    st.write(f'Probability of Delivery Delayed: {prediction_proba[1]:.4f}')
