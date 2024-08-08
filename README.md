# Breast Cancer Prediction Project

The main goal of this project is to develop a machine learning model to predict whether a breast cancer tumor is malignant or benign using data from the Breast Cancer dataset.

### Project Setup
- Created `assignment_4_ann` folder.
- Initialized Git repository.
- Set up virtual environment and installed packages.

### Data Preparation
- Loaded Breast Cancer dataset from sklearn.
- Converted to DataFrame, checked for missing values (none found).
- Split data into training and testing sets.

### Feature Selection
- Used SelectKBest to select top 10 features.
- Standardized selected features.

### Model Tuning
- Performed Grid Search CV to find best hyperparameters for MLPClassifier.

### ANN Model
- Defined and trained a neural network using the best parameters.
- Saved the trained model and scaler.

### Develop a streamlit
- Develop a Streamlit app for interactive model predictions.
