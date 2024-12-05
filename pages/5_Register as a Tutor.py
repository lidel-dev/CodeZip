import streamlit as st

# import db functions
from db3 import create_table, insert_tutors
# Initialize the database (run this only once, ideally at the start of your app)
create_table()

# Main function for the Streamlit app
def main():
    st.title("Register as a Tutor")

    # Input fields for tutor form
    name = st.text_input("Name", key="name_input")
    email = st.text_input("Email", key="email_input")
    phone = st.text_input("Phone Number", key="phone_input")
    message = st.text_area("Message", key="message_input")
    languages = st.text_area("List any Languages/Proficiencies you have in programming.", key="language_input")
    proficiency = st.text_input("What programming language are you most familiar with?", key="optimal_lang_input")
    schedule = st.text_input("Preferred Schedule", key="schedule_input")
    schedule_format = st.text_input("Schedule Format", key="schedule_format_input")
    preferred_schedule = st.text_input("Preferred Schedule Time", key="schedule_time")


    # Submit button to save the data
    if st.button("Submit"):
        if name and email and phone and message and languages and proficiency and schedule and schedule_format and preferred_schedule:
            # Save the tutor info to the database
            insert_tutors(name, email, phone, message, languages, proficiency, schedule, schedule_format, preferred_schedule)
            st.success("Thank you! Your info has been saved.")
        else:
            st.error("Please fill out all fields.")

if __name__ == "__main__":
    main()
