import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import io

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(url, header=None, names=column_names)
    

data.to_csv('iris.csv')

st.title('Iris Data')
st.write('View Iris Data or Download the data')

col1, col2 = st.columns(2)

if col1.checkbox('show raw data'):
    st.subheader('Iris Data')
    st.dataframe(data)

    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    col2.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="iris_data.csv",
        mime="text/csv"
    )


st.title('Fining the Mean,mode and Median of Iris Data')
col1, col2, col3 = st.columns(3)

if col1.button('Mean'):
      
      st.subheader("The Average of Sepal Length per Species")
      avg_sepal_length = data.groupby("species")["sepal_length"].mean()
      st.write(avg_sepal_length)
if col2.button('Median'):
      
      st.subheader("Median of Sepal Length per Species")
      avg_sepal_length = data.groupby("species")["sepal_length"].median()
      st.write(avg_sepal_length)

if col3.button('Largest Value'):
      
      st.subheader("The Average of Sepal Length per Species")
      avg_sepal_length = data.groupby("species")["sepal_length"].idxmax()
      st.write(avg_sepal_length)


st.subheader("Compare two features using a scatter plot")
feature_1 = st.selectbox("Select the first feature:", data.columns[:-1])
feature_2 = st.selectbox("Select the second feature:", data.columns[:-1])

scatter_plot = px.scatter(data, x=feature_1, y=feature_2, color="species", hover_name="species")
st.plotly_chart(scatter_plot)


col1, col2, col3 = st.columns(3)

st.subheader("Filter data based on species")
selected_species = col1.multiselect("Select species to display:", data["species"].unique())

if selected_species:
    filtered_data = data[data["species"].isin(selected_species)]
    st.dataframe(filtered_data)
    
else:
    st.write("No species selected.")


if st.checkbox("Show pairplot for the selected species"):
    st.subheader("Pairplot for the Selected Species")

    if selected_species:
        sns.pairplot(filtered_data, hue="species")
    else:
        sns.pairplot(data, hue="species")
        
    st.pyplot()

st.subheader("Distribution of a Selected Feature")
selected_feature = st.selectbox("Select a feature to display its distribution:", data.columns[:-1])

hist_plot = px.histogram(data, x=selected_feature, color="species", nbins=30, marginal="box", hover_data=data.columns)
st.plotly_chart(hist_plot)