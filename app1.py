import os
import requests
from bs4 import BeautifulSoup
import streamlit as st

# ✅ LangChain v0.2+ imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # type: ignore
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# ✅ Set your Gemini API key

os.environ["GOOGLE_API_KEY"] = "AIzaSyCvWiNdqYt-Du45TQT9W4RRInPx2lCv6LQ"  

# ✅ Your Netlify website link
website_url = "https://sridharshinis.netlify.app/"

def scrape_website(url):
    """Scrape visible text from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = " ".join(
            [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])]
        )
        return page_text
    except Exception as e:
        return f"Error scraping website: {e}"

# ✅ Scrape website + add bio
website_content = scrape_website(website_url)
sridharshini_bio = """
Sridharshini is a pre-final year engineering student passionate about AI and machine learning.
She explores current technologies and trends, building projects that showcase her skills.
Her portfolio is available on her Netlify site.
"""

docs = [
    Document(page_content=sridharshini_bio),
    Document(page_content=website_content)
]

# ✅ Create embeddings and vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever()

# ✅ Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ✅ Streamlit UI
st.title("🤖 Sridharshini AI Chatbot")
st.write("Ask me anything about Sridharshini and her portfolio!")

user_query = st.text_input("Type your question here:")

if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            result = qa_chain(user_query)
