import sqlite3
import pandas as pd
import streamlit as st
import time

# Title
st.title("View All Mentees")

# Path to the SQLite database
db_path = "tutoring_client_info.db"

# Function to fetch data from the database
# Cache data for 10 sec
@st.cache_data(ttl=10)
def fetch_data():
    try:
        conn = sqlite3.connect(db_path)
        # Grab all from tutoring_client_info and connect then retrun the df
        df = pd.read_sql_query("SELECT * FROM tutoring_client_info", conn)
        conn.close()
        return df
    except Exception as e:
        # throw error if cannot grab it
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

# Title
st.title("Contacts Database Viewer")
st.write("ehl")