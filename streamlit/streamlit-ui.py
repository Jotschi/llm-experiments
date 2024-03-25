import os
import openai 
import streamlit as st
import sys

BASE_URL = "http://localhost:10300/v1"
MODEL = "/models/" + sys.argv[1]

print("Using model: " + MODEL)

st.title(MODEL)
client = openai.OpenAI(
        api_key="***",
        base_url=BASE_URL
    )
chat_mode=False

if "model" not in st.session_state:
    st.session_state["model"] = MODEL

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if chat_mode:
            stream = client.chat.completions.create(
                model=st.session_state["model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            completion = client.completions.create(
                model=st.session_state["model"],
                prompt=prompt,
                echo=False,
                n=2,
                stream=False,
                logprobs=3
            )
            response = st.write(completion.choices[0].text)
            st.session_state.messages.append({"role": "assistant", "content": response})