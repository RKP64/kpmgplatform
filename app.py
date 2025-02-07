import streamlit as st
import pyperclip  # For copying text to clipboard
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()

# Retrieve API keys
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API keys are set
if not groq_api_key or not openai_api_key:
    st.error("API keys are not set. Please check your .env file.")
    st.stop()

# Initialize Streamlit page
st.set_page_config(page_title="Compliance Assistant", layout="wide")

# Custom CSS for Sleek Design
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 14px;
        margin: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .response-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333333;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .follow-up-question {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #e3f2fd;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .copy-button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
    }
    .copy-button:hover {
        background-color: #1565c0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Settings
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Select Model:", ["llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b","mixtral-8x7b-32768", "gemma2-9b-it"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_context_length = st.sidebar.number_input("Max Context Length (tokens):", 1000, 8000, 3000)
retrieve_mode = st.sidebar.selectbox("Retrieve Mode:", ["Text (Hybrid)", "Vector Only", "Text Only"])

# Document upload
st.header("Compliance Assistant")
uploaded_files = st.file_uploader("Upload PDF(s):", type="pdf", accept_multiple_files=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Document processing
vector_store = None
if uploaded_files:
    st.subheader("Processing Documents...")
    for uploaded_file in uploaded_files:
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            # Embed chunks into vector store
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            if vector_store is None:
                vector_store = FAISS.from_texts(chunks, embeddings)
            else:
                temp_vector_store = FAISS.from_texts(chunks, embeddings)
                vector_store.merge_from(temp_vector_store)

            st.success(f"Processed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Ask a question
st.header("Ask your Assistant")
predefined_questions = [
    "Verify if Bank of Baroda directly credits loans to borrower accounts, ensuring no unauthorized third-party involvement.",
    "Check if APR, fees, and penalties are fully disclosed upfront, with no hidden charges, and proper grievance redressal exists.",
    "Ensure borrower data is stored in India, biometric data is not retained, and LSPs follow RBI’s data privacy rules.",
]
question = st.radio("Choose a predefined question or type your own:", predefined_questions)
custom_question = st.text_input("Or type your custom question:")

if st.button("Submit"):
    if custom_question:
        question = custom_question

    if vector_store and question:
        # Retrieve context
        relevant_chunks = vector_store.similarity_search(question, k=3)
        context = " ".join([chunk.page_content for chunk in relevant_chunks])

        if len(context) > max_context_length:
            context = context[:max_context_length]

        # Generate response
        try:
            system_message = {
                "role": "system",
                "content": "You are a compliance officer at Bank of Baroda. Provide precise and concise answers to Mr. Pandey based on the provided context. Ensure factual accuracy and include references where applicable."
            }
            user_message = {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer concisely and include references with follow-up questions."
                )
            }
            llm = ChatGroq(model_name=selected_model, api_key=groq_api_key)
            response = llm.invoke([system_message, user_message], temperature=temperature)
            response_text = response.content

            # Ensure response addresses Mr. Pandey
            if not response_text.strip().startswith("Hi Mr. Pandey"):
                response_text = f"Hi Mr. Pandey, {response_text.strip()}"

            # Extract follow-up questions
            follow_up_questions = []
            if "Follow-up questions:" in response_text:
                split_response = response_text.split("Follow-up questions:")
                main_response = split_response[0]
                follow_up_questions = [q for q in split_response[1].strip().split("\n") if q.strip() and not q.lower().startswith("references")]
            else:
                main_response = response_text

            # Display response
            st.markdown(f"<div class='response-box'><b>Response:</b><br>{main_response}</div>", unsafe_allow_html=True)

            # Display follow-up questions with "Copy" buttons
            if follow_up_questions:
                st.markdown("<b>Follow-up Questions:</b>", unsafe_allow_html=True)
                for idx, follow_up in enumerate(follow_up_questions):
                    follow_up_html = (
                        f"<div class='follow-up-question'>"
                        f"<span>{idx + 1}. {follow_up.strip()}</span>"
                        f"<button class='copy-button' onclick='navigator.clipboard.writeText(`{follow_up.strip()}`)'>Copy</button>"
                        f"</div>"
                    )
                    st.markdown(follow_up_html, unsafe_allow_html=True)

            # Log conversation
            st.session_state.conversation_history.append({"question": question, "response": main_response})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please upload and process a document first.")

# Conversation history
if st.session_state.conversation_history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Q{idx + 1}:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")