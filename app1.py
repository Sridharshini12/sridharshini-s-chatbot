import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Get API key from Streamlit Cloud secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.title("🤖 Sridharshini AI Chatbot")

st.write("Ask me anything about Sridharshini's background, AI projects, and more!")

# ✅ Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# ✅ Chat input
user_query = st.text_input("Your Question:")

if user_query:
    with st.spinner("Thinking..."):
        response = llm.invoke(user_query)
        st.success(response.content)

