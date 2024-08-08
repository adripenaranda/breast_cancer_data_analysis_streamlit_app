import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap  # Import textwrap for wrapping labels

# Load the pre-trained model and scaler from disk
model = load_model('model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Display the main title of the Streamlit app with a specific color
st.markdown("<h1 style='color: #FFA07A;'>Breast Cancer Prediction App</h1>", unsafe_allow_html=True)

# Create a sidebar with input sliders for user input and a colored header
st.sidebar.markdown("<h2 style='color: #20B2AA;'>Input Features</h2>", unsafe_allow_html=True)

# Define a function to capture user input features from the sidebar
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
    
    # Store the input values in a dictionary
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
    
    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Call the function to get user input and store it in a DataFrame
input_df = user_input_features()

# Ensure the order of input features matches the scaler's expected order
selected_features = [
    'mean radius', 'mean perimeter', 'mean area', 'mean concavity', 
    'mean concave points', 'worst radius', 'worst perimeter', 
    'worst area', 'worst concavity', 'worst concave points'
]

# Reorder the input DataFrame columns to match the selected features
input_df = input_df[selected_features]

# Scale the user input features using the loaded scaler
scaled_input = scaler.transform(input_df)

# Make a prediction using the pre-trained model
prediction = model.predict(scaled_input)

# Display the prediction result with appropriate color coding
st.subheader('Prediction')
if prediction[0][0] > 0.5:
    st.markdown("<h3 style='color: #FF4500;'>Malignant</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color: #32CD32;'>Benign</h3>", unsafe_allow_html=True)

# Display the prediction probability
st.subheader('Prediction Probability')
st.write(f"{prediction[0][0] * 100:.2f}%")

# Define a function to wrap text labels for better readability
def wrap_labels(labels, width):
    return [textwrap.fill(label, width) for label in labels]

# Create two columns for side-by-side visualizations
col1, col2 = st.columns(2)

# Select features related to size for visualization
size_features = ['mean radius', 'mean perimeter', 'mean area', 'worst radius', 'worst perimeter', 'worst area']
size_df = input_df[size_features]

# Display a bar plot of size features in the first column
with col1:
    st.subheader('Size Features')
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figsize to make the plot bigger
    sns.barplot(x=size_df.columns, y=size_df.iloc[0], ax=ax, palette='coolwarm')
    ax.set_xticklabels(wrap_labels(['Mean Radius (µm)', 'Mean Perimeter (µm)', 'Mean Area (µm²)', 
                                    'Worst Radius (µm)', 'Worst Perimeter (µm)', 'Worst Area (µm²)'], 10), 
                       rotation=0, horizontalalignment='center', fontsize=14)
    ax.set_ylabel('Values (µm or µm²)', fontsize=16)
    ax.set_xlabel('Features', fontsize=16)
    st.pyplot(fig)

# Select other features for visualization
other_features = ['mean concavity', 'mean concave points', 'worst concavity', 'worst concave points']
other_df = input_df[other_features]

# Display a bar plot of other features in the second column
with col2:
    st.subheader('Other Features')
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figsize to make the plot bigger
    sns.barplot(x=other_df.columns, y=other_df.iloc[0], ax=ax, palette='viridis')
    ax.set_xticklabels(wrap_labels(['Mean Concavity', 'Mean Concave Points', 
                                    'Worst Concavity', 'Worst Concave Points'], 10), 
                       rotation=0, horizontalalignment='center', fontsize=14)
    ax.set_ylabel('Dimensionless Values', fontsize=16)
    ax.set_xlabel('Features', fontsize=16)
    st.pyplot(fig)

# Display a section with descriptions of the input features
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

# Add a feedback section in the sidebar
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Please provide your feedback here:")
if st.sidebar.button("Submit"):
    st.sidebar.write("Thank you for your feedback!")
