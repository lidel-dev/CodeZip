import streamlit as st

# link to db2 file
from db2 import create_table, insert_contact

# first function initialized
create_table()

def main():
    st.title("Tutoring Service Sign-up Form")

    # Input fields for tutoring sign-up form
    name = st.text_input("Name", key="name_input")
    email = st.text_input("Email", key="email_input")
    phone = st.text_input("Phone Number", key="phone_input")
    language_improve = st.text_input("Language to Improve", key="language_input")
    skill_level = st.text_input("Skill Level", key="skill_input")
    focus = st.text_input("Focus Areas", key="focus_input")
    goals = st.text_input("Goals", key="goals_input")
    schedule = st.text_input("Preferred Schedule", key="schedule_input")
    schedule_format = st.text_input("Schedule Format", key="schedule_format_input")
    preferred_schedule = st.text_input("Preferred Schedule Time", key="preferred_schedule_input")
    tutoring_amount = st.text_input("Amount per Session", key="tutoring_amount_input")
    tutoring_format = st.text_input("Preferred Tutoring Format", key="tutoring_format_input")
    payment_method = st.text_input("Payment Method", key="payment_method_input")
    billing_address = st.text_input("Billing Address", key="billing_address_input")
    terms_agreement = st.text_input("Agreement to Terms", key="terms_input")
    recording_consent = st.text_input("Consent to Record Sessions", key="recording_consent_input")
    
    # Submit button to save the data
    if st.button("Submit"):
        if name and email and phone and language_improve and skill_level and focus and goals and schedule and schedule_format and preferred_schedule and tutoring_amount and tutoring_format and payment_method and billing_address and terms_agreement and recording_consent:
            # Save the contact info to the database
            insert_contact(name, email, phone, language_improve, skill_level, focus, goals, schedule, schedule_format, preferred_schedule, tutoring_amount, tutoring_format, payment_method, billing_address, terms_agreement, recording_consent)
            st.success("Your contact info has been saved.")
        else:
            st.error("Please fill out all fields.")

if __name__ == "__main__":
    main()
