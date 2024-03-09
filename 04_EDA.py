import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv('crypto_data_reduced.csv')
df = df[['Index', 'Cluster']]

# Widget to filter data based on 'Index'
index_to_filter = st.sidebar.multiselect('Select Index:', df['Index'].unique())

# Filter the DataFrame based on the user's selection for 'Index'
if index_to_filter:
    df = df[df['Index'].isin(index_to_filter)]

# Widget to filter data based on 'Cluster'
cluster_to_filter = st.sidebar.multiselect('Select cluster:', df['Cluster'].unique())

# Filter the DataFrame based on the user's selection for 'Cluster'
if cluster_to_filter:
    df = df[df['Cluster'].isin(cluster_to_filter)]

# Display the DataFrame
st.write(df)
