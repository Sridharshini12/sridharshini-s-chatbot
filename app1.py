import os
import requests
from bs4 import BeautifulSoup
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# ✅ Load Gemini API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ✅ Your portfolio site
portfolio_url = "https://sridharshinis.netlify.app/"

def scrape_website(url):
    """Scrape visible text from a webpage"""
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return " ".join(
            [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "li"])]
        )
    except Exception as e:
        return f"Error scraping website: {e}"

# ✅ Combine scraped portfolio + short bio
portfolio_text = scrape_website(portfolio_url)
bio_text = """
Sridharshini is a pre-final year engineering student passionate about AI & ML. 
She builds projects, explores trends, and showcases them in her portfolio.
"""

docs = [
    Document(page_content=bio_text),
    Document(page_content=portfolio_text)
]

# ✅ Create embeddings + vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever()

# ✅ Chat model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# ✅ Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# ✅ Streamlit UI
st.title("🤖 Sridharshini AI Chatbot")
st.write("Ask me about Sridharshini and her portfolio!")

query = st.text_input("Your Question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                result = qa_chain(query)
                st.subheader("💡 Answer:")
                st.write(result["result"])
            except Exception as e:
                st.error("⚠️ Gemini API Error. Check your key or model.")
                st.code(str(e))
    else:
        st.warning("Please enter a question!")
