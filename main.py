import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from persona_config import MODI_PROMPT, TRUMP_PROMPT

load_dotenv()
CHROMA_PATH = "chroma"

st.title("PDC RAG Chatbot: Modi vs Trump")
persona = st.radio("Choose Leader:", ["Narendra Modi", "Donald Trump"], horizontal=True)
persona_type = "modi" if "Modi" in persona else "trump"

if persona_type == "modi":
    st.success("Currently: Modi Mode")
else:
    st.error("Currently: Trump Mode")

@st.cache_resource
def setup_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", k=5)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"]
    )
    return retriever, llm

try:
    retriever, llm = setup_rag()
except Exception as e:
    st.error(f"Error: {e}")
    st.info("Run python create_db.py first and add GROQ_API_KEY to .env")
    st.stop()

def answer_with_persona(query_text):
    persona_prompt = MODI_PROMPT if persona_type == "modi" else TRUMP_PROMPT
    system_prompt = persona_prompt + "\n\nContext from PDC Course:\n{context}"
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, chain)
    return rag_chain.invoke({"input": query_text})["answer"]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about PDC course...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        try:
            response = answer_with_persona(query)
        except Exception as e:
            response = f"Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(f"**{persona_type.capitalize()} Mode:**")
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("About")
    st.info("RAG chatbot using LangChain + ChromaDB + Groq")