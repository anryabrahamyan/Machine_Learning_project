"""
File for hosting the streamlit app
"""
from typing import List
import streamlit as st
from model_app.utils import *

# streamlit
#Page Title
st.set_page_config(page_title="Emotion Prediction From Tweets", initial_sidebar_state="collapsed")

#
input_text = st.text_input(
    "Input Your Text",
    value="Machine learning is fun",
)

predictions = predict(input_text)

st.write('We predicted:', predict(input_text))


if __name__ == "__main__":
    pass
