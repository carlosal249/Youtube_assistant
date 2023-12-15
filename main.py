import streamlit as st
import langchain_hellper as lch
import textwrap

st.title('youtube-assistant')

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label='input your youtube video URL',
            max_chars=50
        )

        query = st.sidebar.text_area(
            label='ask me about the video',
            max_chars=50,
            key='query'
        )

        submit_btn = st.form_submit_button(label='Submit')

if query and youtube_url:
    db = lch.create_vector_from_yt(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    st.subheader('LLM Answer:')
    st.text(
        textwrap.fill(response, width=80)
    )

