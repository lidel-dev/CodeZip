import sqlite3
import pandas as pd
import streamlit as st
import time

# Title
st.title("View All Tutors")

# Path to the SQLite database
db_path = "tutors.db"

# Function to fetch data from the database
@st.cache_data(ttl=10)  # Cache the data for 10 seconds
def fetch_data():
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM tutors", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# Display the DataFrame and update automatically every 10 seconds
placeholder = st.empty() 

# If true call the function and display the data
while True:
    data = fetch_data()
    if not data.empty:
        placeholder.dataframe(data)
    else:
        placeholder.warning("No data returned. Check your database.")
    time.sleep(10)  # Refresh every 10 seconds
