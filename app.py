import streamlit as st
import pandas as pd
import data_loader
import visualizations
import regression_models

# Load data
data = data_loader.load_data()

# Title and description
title_html = """
    <div style="background-image: linear-gradient(to right, #FFA500, #008080);
                border-radius: 30%; padding: 20px; text-align: center;">
        <h1 style="color: white; font-style: italic; font-size: 5.5em;">AtmosAware</h1>
    </div>
"""

st.markdown(title_html, unsafe_allow_html=True)
st.title('Air Quality Analysis App:')
st.write("This app provides an interactive platform for exploring comprehensive air quality data, conducting sophisticated regression analysis, visualizing trends, and generating insightful predictions to understand and analyze the complex relationships between various air quality parameters.")

# Display data
if st.checkbox('Show Data'):
    st.subheader('Data')
    st.write(data)

# Visualizations
visualizations.display_pm25_time_plot(data)
visualizations.display_pm25_3d_plot(data)

# Sidebar for Regression
# st.sidebar.title("Welcome to the App:")

st.sidebar.title("Welcome to the App:")
st.sidebar.write("This app explores air quality data and performs regression analysis.")
st.sidebar.title("Regression Models:")

# Regression model
selected_model = st.sidebar.selectbox("Select Regression Model", ['Random Forest', 'Linear Regression', 'Support Vector Machine','Lasso Regression','Ridge Regression','Decision Tree','Bayesian Regression'])

if selected_model == 'Random Forest':
    regression_models.random_forest_regression(data)
elif selected_model == 'Linear Regression':
    regression_models.linear_regression(data)
elif selected_model == 'Support Vector Machine':
    regression_models.support_vector_machine(data)
elif selected_model == 'Lasso Regression':
    regression_models.lasso_regression(data)
elif selected_model == 'Ridge Regression':
    regression_models.ridge_regression(data)
elif selected_model == 'Decision Tree':
    regression_models.decision_tree_regression(data)
elif selected_model == 'Bayesian Regression':
    regression_models.bayesian_regression(data)
    # Add more models as needed in a similar manner
