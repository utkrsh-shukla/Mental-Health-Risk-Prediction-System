import streamlit as st
# Path to the background image
image_path = 'Jiraya.png'

# HTML/CSS for background image
background_style = f"""
<style>
    .stApp {{
        background-image: url("{image_path}");
        background-size: cover;
        background-position: center;
        height: 100vh;  /* Full screen height */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}
</style>
"""

# Apply the background style
st.markdown(background_style, unsafe_allow_html=True)


st.title("🧠 Mental-Health-Risk-Prediction-System")
st.write(
    "On it , will be done within 3 days"
)
