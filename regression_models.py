import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def random_forest_regression(data):
    st.subheader("Random Forest Regression")
    
    X = data[['Year', 'Month', 'Day', 'Hour']]
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_rf = RandomForestRegressor()
    reg_rf.fit(X_train, y_train)

    y_pred = reg_rf.predict(X_test)

    st.write("Evaluation Metrics")
    st.write(f"R-squared: {reg_rf.score(X_test, y_test)}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    st.write("### Actual vs Predicted PM2.5 values")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    st.pyplot(plt)

    target_column = 'PM2.5'

    # Assume other columns are features
    features = [col for col in data.columns if col != target_column and data[col].dtype != 'object']


    # Assume X contains features, and y contains the target variable
    X = data[features]
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R2 Score: {r2}")

    # Prediction with user input
    st.header('Predict PM2.5 Value')

    # User input for feature values
    user_input = {}
    for feature in features:
        if feature == 'Year':
            user_input[feature] = st.slider(f"Enter value for {feature}", 2000, 2100, 2050, format='%d')
        # user_input[feature] = st.slider(f"Enter value for {feature}", int(data[feature].min()), int(data[feature].max()), int(data[feature].mean()))
        elif data[feature].dtype == 'int64':
            user_input[feature] = st.slider(f"Enter value for {feature}", int(data[feature].min()), int(data[feature].max()), int(data[feature].mean()), format='%d')
        else:
            user_input[feature] = st.slider(f"Enter value for {feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))


    # Convert user input to DataFrame for prediction
    user_df = pd.DataFrame([user_input])

    # Predict PM2.5 based on user input
    predicted_pm25 = model.predict(user_df)

    st.write(f"Predicted PM2.5 value: {predicted_pm25[0]}")

def linear_regression(data):
    st.subheader("Linear Regression")
    X = data[['Year', 'Month', 'Day', 'Hour']]
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    st.write("Evaluation Metrics")
    st.write(f"R-squared: {reg.score(X_test, y_test)}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    st.write("### Actual vs Predicted PM2.5 values")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    st.pyplot(plt)

def support_vector_machine(data):
    st.subheader("Support Vector Machine")
    X = data[['Year', 'Month', 'Day', 'Hour']]
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = SVR()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    st.write("Evaluation Metrics")
    st.write(f"R-squared: {reg.score(X_test, y_test)}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    st.write("### Actual vs Predicted PM2.5 values")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    st.pyplot(plt)

def lasso_regression(data):
    st.subheader("Lasso Regression")
    X = data[['Year', 'Month', 'Day', 'Hour']]
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = Lasso()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    st.write("Evaluation Metrics")
    st.write(f"R-squared: {reg.score(X_test, y_test)}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    st.write("### Actual vs Predicted PM2.5 values")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    st.pyplot(plt)

def ridge_regression(data):
    st.subheader("Ridge Regression")
    X = data[['Year', 'Month', 'Day', 'Hour']]
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = Ridge()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    st.write("Evaluation Metrics")
    st.write(f"R-squared: {reg.score(X_test, y_test)}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    st.write("### Actual vs Predicted PM2.5 values")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    st.pyplot(plt)

def decision_tree_regression(data):
    st.subheader("Decision Tree Regression")
    X = data[['Year', 'Month', 'Day', 'Hour']]
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    st.write("Evaluation Metrics")
    st.write(f"R-squared: {reg.score(X_test, y_test)}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    st.write("### Actual vs Predicted PM2.5 values")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    st.pyplot(plt)

def bayesian_regression(data):
    st.subheader("Bayesian Ridge Regression")
    X = data[['Year', 'Month', 'Day', 'Hour']]
    y = data['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = BayesianRidge()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    st.write("Evaluation Metrics")
    st.write(f"R-squared: {reg.score(X_test, y_test)}")
    st.write(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

    st.write("### Actual vs Predicted PM2.5 values")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    st.pyplot(plt)