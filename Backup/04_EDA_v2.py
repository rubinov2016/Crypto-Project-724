import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv('crypto_data_reduced.csv')
df = df[['Index', 'Cluster']]


# Filtering UI
st.write("### Filter Options:")
stock_filter = st.selectbox('Filter by Stock:', options=[''] + list(df['Index'].unique()))
cluster_filter = st.selectbox('Filter by Cluster:', options=[''] + list(df['Cluster'].unique()))

# Apply filters if selected
if stock_filter:
    df = df[df['Index'] == stock_filter]
if cluster_filter:
    df = df[df['Cluster'] == cluster_filter]

# Sorting UI
sort_by = st.selectbox('Sort by:', options=['', 'Index', 'Cluster'])
if sort_by:
    df = df.sort_values(by=sort_by)

# Display the DataFrame
st.write("### Filtered and Sorted Data:")
st.dataframe(df)
