import streamlit as st
import pandas as pd

@st.cache
def load_data():
    data = pd.read_csv("air-quality-india.csv")
    # Add data preprocessing steps here if needed
    return data
