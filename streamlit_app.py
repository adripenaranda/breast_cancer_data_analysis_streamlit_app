import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
model = load_model('model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app title
st.title("Breast Cancer Prediction App")

# Sidebar for user inputs
st.sidebar.header("Input Features")

def user_input_features():
    mean_radius = st.sidebar.slider("Mean Radius", 6.0, 30.0, 14.0)
    mean_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 200.0, 90.0)
    mean_area = st.sidebar.slider("Mean Area", 140.0, 2500.0, 700.0)
    mean_concavity = st.sidebar.slider("Mean Concavity", 0.0, 0.5, 0.1)
    mean_concave_points = st.sidebar.slider("Mean Concave Points", 0.0, 0.3, 0.1)
    worst_radius = st.sidebar.slider("Worst Radius", 10.0, 40.0, 20.0)
    worst_perimeter = st.sidebar.slider("Worst Perimeter", 50.0, 300.0, 140.0)
    worst_area = st.sidebar.slider("Worst Area", 200.0, 5000.0, 1500.0)
    worst_concavity = st.sidebar.slider("Worst Concavity", 0.0, 1.5, 0.5)
    worst_concave_points = st.sidebar.slider("Worst Concave Points", 0.0, 0.5, 0.2)
    
    data = {
        'mean radius': mean_radius,
        'mean perimeter': mean_perimeter,
        'mean area': mean_area,
        'mean concavity': mean_concavity,
        'mean concave points': mean_concave_points,
        'worst radius': worst_radius,
        'worst perimeter': worst_perimeter,
        'worst area': worst_area,
        'worst concavity': worst_concavity,
        'worst concave points': worst_concave_points
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure the input features match the scaler's expected order
selected_features = [
    'mean radius', 'mean perimeter', 'mean area', 'mean concavity', 
    'mean concave points', 'worst radius', 'worst perimeter', 
    'worst area', 'worst concavity', 'worst concave points'
]

input_df = input_df[selected_features]

# Scale the input features
scaled_input = scaler.transform(input_df)

# Make prediction
prediction = model.predict(scaled_input)

# Output prediction
st.subheader('Prediction')
if prediction[0][0] > 0.5:
    st.write("Malignant")
else:
    st.write("Benign")

st.subheader('Prediction Probability')
st.write(f"{prediction[0][0] * 100:.2f}%")

# Visualization of Input Features
st.subheader('Input Features Visualization')
st.write("The following chart shows the input features you provided:")
fig, ax = plt.subplots()
sns.barplot(x=input_df.columns, y=input_df.iloc[0], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
st.pyplot(fig)

# Education section
st.subheader('Understanding the Features')
st.write("""
- **Mean Radius**: Average of distances from center to points on the perimeter.
- **Mean Perimeter**: Perimeter of the tumor.
- **Mean Area**: Area of the tumor.
- **Mean Concavity**: Severity of concave portions of the contour.
- **Mean Concave Points**: Number of concave portions of the contour.
- **Worst Radius**: Largest distance from center to points on the perimeter.
- **Worst Perimeter**: Largest perimeter of the tumor.
- **Worst Area**: Largest area of the tumor.
- **Worst Concavity**: Largest severity of concave portions of the contour.
- **Worst Concave Points**: Largest number of concave portions of the contour.
""")

# Additional interactivity: User feedback
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Please provide your feedback here:")
if st.sidebar.button("Submit"):
    st.sidebar.write("Thank you for your feedback!")

# Display dataset statistics (Optional for user education)
st.subheader('Dataset Statistics')
if st.checkbox('Show statistics'):
    st.write("Here are some statistics of the dataset used to train the model:")
