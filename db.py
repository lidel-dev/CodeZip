# NOTE: This is the file for contacts.db

import sqlite3

# Create the contacts table
def create_table():
    conn = sqlite3.connect('contacts.db')  # Connect to SQLite database
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

# Insert a new contact into the database
def insert_contact(name, email, phone, message):
    conn = sqlite3.connect('contacts.db')
    c = conn.cursor()
    c.execute("INSERT INTO contacts (name, email, phone, message) VALUES (?, ?, ?, ?)",
            (name, email, phone, message))
    conn.commit()
    conn.close()
