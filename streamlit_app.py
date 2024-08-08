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

# Streamlit app title with color
st.markdown("<h1 style='color: #FFA07A;'>Breast Cancer Prediction App</h1>", unsafe_allow_html=True)

# Sidebar for user inputs with header color
st.sidebar.markdown("<h2 style='color: #20B2AA;'>Input Features</h2>", unsafe_allow_html=True)

def user_input_features():
    mean_radius = st.sidebar.slider("Mean Radius (µm)", 6.0, 30.0, 14.0)
    mean_perimeter = st.sidebar.slider("Mean Perimeter (µm)", 40.0, 200.0, 90.0)
    mean_area = st.sidebar.slider("Mean Area (µm²)", 140.0, 2500.0, 700.0)
    mean_concavity = st.sidebar.slider("Mean Concavity", 0.0, 0.5, 0.1)
    mean_concave_points = st.sidebar.slider("Mean Concave Points", 0.0, 0.3, 0.1)
    worst_radius = st.sidebar.slider("Worst Radius (µm)", 10.0, 40.0, 20.0)
    worst_perimeter = st.sidebar.slider("Worst Perimeter (µm)", 50.0, 300.0, 140.0)
    worst_area = st.sidebar.slider("Worst Area (µm²)", 200.0, 5000.0, 1500.0)
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

# Output prediction with color
st.subheader('Prediction')
if prediction[0][0] > 0.5:
    st.markdown("<h3 style='color: #FF4500;'>Malignant</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color: #32CD32;'>Benign</h3>", unsafe_allow_html=True)

st.subheader('Prediction Probability')
st.write(f"{prediction[0][0] * 100:.2f}%")

# Create two columns for side-by-side visualizations
col1, col2 = st.columns(2)

# Visualization for size features
size_features = ['mean radius', 'mean perimeter', 'mean area', 'worst radius', 'worst perimeter', 'worst area']
size_df = input_df[size_features]

with col1:
    st.subheader('Size Features Visualization')
    fig, ax = plt.subplots()
    sns.barplot(x=size_df.columns, y=size_df.iloc[0], ax=ax, palette='coolwarm')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel('Values (µm or µm²)')
    ax.set_xlabel('Features')
    st.pyplot(fig)

# Visualization for other features
other_features = ['mean concavity', 'mean concave points', 'worst concavity', 'worst concave points']
other_df = input_df[other_features]

with col2:
    st.subheader('Other Features Visualization')
    fig, ax = plt.subplots()
    sns.barplot(x=other_df.columns, y=other_df.iloc[0], ax=ax, palette='viridis')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel('Dimensionless Values')
    ax.set_xlabel('Features')
    st.pyplot(fig)

# Education section with colored headers
st.subheader('Understanding the Features')
st.markdown("""
<ul>
    <li><b>Mean Radius</b>: Average of distances from center to points on the perimeter (µm).</li>
    <li><b>Mean Perimeter</b>: Perimeter of the tumor (µm).</li>
    <li><b>Mean Area</b>: Area of the tumor (µm²).</li>
    <li><b>Mean Concavity</b>: Severity of concave portions of the contour (dimensionless).</li>
    <li><b>Mean Concave Points</b>: Number of concave portions of the contour (dimensionless).</li>
    <li><b>Worst Radius</b>: Largest distance from center to points on the perimeter (µm).</li>
    <li><b>Worst Perimeter</b>: Largest perimeter of the tumor (µm).</li>
    <li><b>Worst Area</b>: Largest area of the tumor (µm²).</li>
    <li><b>Worst Concavity</b>: Largest severity of concave portions of the contour (dimensionless).</li>
    <li><b>Worst Concave Points</b>: Largest number of concave portions of the contour (dimensionless).</li>
</ul>
""", unsafe_allow_html=True)

# Additional interactivity: User feedback
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Please provide your feedback here:")
if st.sidebar.button("Submit"):
    st.sidebar.write("Thank you for your feedback!")

