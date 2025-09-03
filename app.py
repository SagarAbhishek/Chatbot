import streamlit as st
from chatbot_core import process_and_store_pdf, create_qa_chain
import os

# --- CONFIGURATION ---
# 1. SET YOUR PDF PATH
#    Place your PDF in the same folder as this script and update the name here.
PDF_PATH = "classifier.pdf"  # <--- IMPORTANT: CHANGE THIS TO YOUR PDF'S FILENAME

# --- APP LAYOUT ---
st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("Local PDF Chatbot 💬")
st.markdown("---")

# --- CORE LOGIC ---
# Initialize the QA chain once and store it in the session state
if "qa_chain" not in st.session_state:
    if os.path.exists(PDF_PATH):
        with st.spinner(f"Processing '{PDF_PATH}'... This may take a moment on first run."):
            try:
                vectorstore = process_and_store_pdf(PDF_PATH)
                st.session_state.qa_chain = create_qa_chain(vectorstore)
                st.success(f"Successfully processed '{PDF_PATH}'. Ready to chat!")
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {e}")
                st.session_state.qa_chain = None
    else:
        st.error(f"Error: The file '{PDF_PATH}' was not found. Please make sure it's in the same folder as the app.")
        st.session_state.qa_chain = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask a question about your PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        if st.session_state.qa_chain:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(prompt)
                st.markdown(response)
        else:
            response = "The QA chain is not initialized. Please check the file path and restart."
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})