# NOTE: This is the file for tutoring_client_info.db

import sqlite3



# Create the contacts table
def create_table():
    conn = sqlite3.connect('tutoring_client_info.db')  # Connect to SQLite database
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tutoring_client_info (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            
            language_improve TEXT,
            skill_level TEXT,
            focus TEXT,
            goals TEXT,
            
            schedule TEXT,
            schedule_format TEXT,
            preferred_schedule TEXT,
            
            tutoring_amount TEXT,
            tutoring_format TEXT,
            payment_method TEXT,
            billing_address TEXT,
            terms_agreement TEXT,
            recording_consent TEXT
        )
    """)
    conn.commit()
    conn.close()

# Insert a new contact into the database
def insert_contact(name, email, phone, language_improve, skill_level, focus, goals, schedule, schedule_format, preferred_schedule, tutoring_amount, tutoring_format, payment_method, billing_address, terms_agreement, recording_consent):
    conn = sqlite3.connect('tutoring_client_info.db')
    c = conn.cursor()
    c.execute("INSERT INTO tutoring_client_info (name, email, phone, language_improve, skill_level, focus, goals, schedule, schedule_format, preferred_schedule, tutoring_amount, tutoring_format, payment_method, billing_address, terms_agreement, recording_consent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, email, phone, language_improve, skill_level, focus, goals, schedule, schedule_format, preferred_schedule, tutoring_amount, tutoring_format, payment_method, billing_address, terms_agreement, recording_consent))
    conn.commit()
    conn.close()
