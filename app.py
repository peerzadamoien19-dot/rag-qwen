import streamlit as st

from ingest import read_pdf, make_chunks
from indexer import build_index
from retriever import search
from generator import ask_qwen

st.set_page_config(page_title="RAG with Qwen", page_icon="📄")
st.title("Chat with your PDF")


@st.cache_resource(show_spinner=False)
def prepare(file):
    text = read_pdf(file)
    chunks = make_chunks(text)
    index = build_index(chunks)
    return index, chunks


with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

with st.spinner("Reading and indexing your PDF..."):
    index, chunks = prepare(uploaded)

st.sidebar.success(f"Ready. {len(chunks)} chunks indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.write(s)
                    st.divider()

question = st.chat_input("Ask something about your PDF")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            found = search(question, index, chunks)
            answer = ask_qwen(question, found, st.secrets["HF_TOKEN"])

        st.write(answer)

        with st.expander("Sources used"):
            for s in found:
                st.write(s)
                st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": found}
    )