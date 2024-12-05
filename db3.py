# NOTE: This is the file for tutors.db (info on signed up tutors)

import sqlite3

# Create the tutors table
def create_table():
    conn = sqlite3.connect('tutors.db')  # Connect to SQLite database
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tutors (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            message TEXT,
            languages TEXT,
            proficiency TEXT,
            schedule TEXT,
            schedule_format TEXT,
            preferred_schedule TEXT
        )
    """)
    conn.commit()
    conn.close()

# Insert a new tutor into the database
def insert_tutors(name, email, phone, message, languages, proficiency, schedule, schedule_format, preferred_schedule):

    conn = sqlite3.connect("tutors.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO tutors (name, email, phone, message, languages, proficiency, schedule, schedule_format, preferred_schedule)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, email, phone, message, languages, proficiency, schedule, schedule_format, preferred_schedule))

    conn.commit()
    conn.close()
