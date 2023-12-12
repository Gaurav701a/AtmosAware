import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

def display_pm25_time_plot(data):
    st.write("### PM2.5 With Respect to Time")
    fig, ax = plt.subplots()
    data.plot(x='Timestamp', y='PM2.5', figsize=(10, 6), ax=ax)
    plt.xlabel("Timestamp")
    plt.ylabel("Particulate Matter 2.5")
    st.pyplot(fig)

def display_pm25_3d_plot(data):
    st.write("### Distribution of Particulate Matter by Month and Year (3D Scatter Plot)")
    fig_3d_year_month = px.scatter_3d(data, x="Year", y="Month", z="PM2.5", color="PM2.5",
                                      color_continuous_scale=px.colors.sequential.Viridis,
                                      title='PM2.5 by Month and Year')
    st.plotly_chart(fig_3d_year_month)
