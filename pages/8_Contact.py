import streamlit as st
# import db functions
from db import create_table, insert_contact
# Initialize the database (run this only once, ideally at the start of your app)
create_table()

# Main function for the Streamlit app
def main():
    st.title("Contact Us")

    # Input fields for contact form
    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    message = st.text_area("Message")

    # Submit button to save the data
    if st.button("Submit"):
        if name and email and phone and message:
            # Save the contact info to the database
            insert_contact(name, email, phone, message)
            st.success("Thank you! Your contact info has been saved.")
        else:
            st.error("Please fill out all fields.")

if __name__ == "__main__":
    main()
