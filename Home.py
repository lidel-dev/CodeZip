import streamlit as st

st.header('CodeZip: Uniting Programmers Since 2024', divider='rainbow')

# Add in the GIF
st.image("assets/Homepage.gif", use_container_width=True)

st.markdown('')

# Key Features of Python
st.markdown(f'''
## About Our Site
---
Whether you're looking to start coding or searching for a tutoring side
hustle, CodeZip has you covered!
Our platform aims to connect aspiring developers to industry professionals through a customized tutoring service. We also accept donations, you can buy us a 
coffee at the *buy us a coffee* button. Thank you for using CodeZip and
we hope you have a meaningful and enjoyable experience. 
            ''')

st.markdown(f'''
### Getting Started
---
Our site consists of three key sections
            ''')
st.markdown('**1. Learn:** Navigate through the numbered pages to get a basic overview of Python, explore some foundational libraries, and gain a deeper understanding of frameworks to ensure you start off the tutoring session right.')
st.markdown('**2. Registration:** Should you be interested in our tutoring service, either as someone looking to learn coding, or as a tutor in search of a client, navigate to the *Register as a Tutor* page or *Tutoring Sign Up* page.')
st.markdown('**3. Get Matched:** After signing up as either a tutor or mentee you can view the *View Tutors* or *View Mentees* page to view sign up information and get in contact.')
st.markdown('**4. Contact Us:** Lastly, refer to the *Contact* page if you have any questions for our team here at CodeZip.')

st.markdown(f'''
#### Support the Further Development of CodeZip

            ''')
st.markdown(
    # Add the 'buy me a coffee' button
    """
    <style>
    .bmc-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 15px;
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        font-weight: 500;
        color: #000000;
        background-color: #FFFFFF;
        border-radius: 5px;
        border: none;
        text-decoration: none;
        cursor: pointer;
        margin-top: 20px;
    }
    .bmc-button img {
        height: 20px;
        margin-right: 8px;
    }
    </style>
    <a class="bmc-button" href="https://buymeacoffee.com/lideldev" target="_blank">
        <img src="https://cdn.buymeacoffee.com/buttons/bmc-new-btn-logo.svg" alt="Buy me a coffee">
        Buy me a coffee
    </a>
    """,
    unsafe_allow_html=True
)



