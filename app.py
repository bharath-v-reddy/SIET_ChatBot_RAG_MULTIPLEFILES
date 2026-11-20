
import streamlit as st
from pathlib import Path
from rag_engine import load_base, ask_rag

st.set_page_config(page_title="SIET_Chatbot", layout="centered")

st.title("Shridevi Institute of Engineering and Technology - ChatBot")
st.caption("Ask anything about college information, Placement, FAQS, Rules and Regulations")
st.caption("RAG pipeline with multiple files")
st.caption("Created by BHARATH V REDDY")

# Load API key
mykey = st.secrets.get("GROQ_API_KEY")

if not mykey:
    st.error("Groq API key is missing. Add it in Streamlit Secrets.")
    st.stop()


# Initialize RAG engine
@st.cache_resource
def initiate_rag():
    pdf_path = "college_docs.pdf"

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    return load_base(pdf_path)


# Load index, embedding model and chunks
try:
    index, embed_model, chunks = initiate_rag()

except Exception as e:
    st.error(f"Error loading RAG system: {e}")
    st.stop()


# Initialize chat history
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []


# Display previous chat messages
for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])


# Accept user question
question = st.chat_input("Ask anything about college....")


if question:

    # Store user message
    st.session_state.chat_hist.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Asking RAG for answer...."):

            try:
                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=mykey
                )

                st.write(answer)

                # Store assistant response
                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(f"Error generating answer: {e}")


# Clear chat history
btn = st.button("Clear Chat")

if btn:
    st.session_state.chat_hist = []
    st.success("Chat History Deleted Successfully")
    st.rerun()
