import streamlit as st
import os
from typing import List, Dict
import tempfile
from datetime import datetime
from operator import itemgetter
import uuid
import hashlib

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from pinecone import Pinecone, ServerlessSpec


st.set_page_config(page_title="Personalized RAG Chat Assistant", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "retrieval_scores" not in st.session_state:
    st.session_state.retrieval_scores = []
if "pinecone_namespace" not in st.session_state:
    st.session_state.pinecone_namespace = str(uuid.uuid4())
if "user_query_history" not in st.session_state:
    st.session_state.user_query_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# API Keys
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_INDEX_NAME = "rag-documents"

def get_file_id(uploaded_file) -> str:
    content = uploaded_file.getvalue()
    return hashlib.md5(content).hexdigest()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return pc

def load_document(file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    
    try:
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith('.txt'):
            loader = TextLoader(tmp_path)
        elif file.name.endswith('.md'):
            loader = TextLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {file.name}")
        
        documents = loader.load()
        return documents
    finally:
        os.unlink(tmp_path)

def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks: List[Document], embeddings, namespace: str):
    try:
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        pc = Pinecone(api_key=PINECONE_API_KEY)
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace,
            pinecone_api_key=PINECONE_API_KEY)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def calculate_retrieval_accuracy(query: str, retrieved_docs: List[Document]) -> Dict:
    if not retrieved_docs:
        return {"num_docs": 0, "avg_score": 0.0, "relevance": "No documents retrieved"}
    
    query_words = set(query.lower().split())
    relevance_scores = []
    
    for doc in retrieved_docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words.intersection(doc_words))
        score = overlap / len(query_words) if query_words else 0.5
        relevance_scores.append(score+0.2)
    
    avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
    
    return {
        "num_docs": len(retrieved_docs),
        "avg_score": round(avg_score, 3),
        "relevance": "High" if avg_score > 0.5 else "Medium" if avg_score > 0.2 else "Low",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(session_id: str):
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = ChatMessageHistory()
    return st.session_state.lc_history

def build_retrieval_query(current_question: str, window_size: int = 3) -> str:
    history = st.session_state.get("user_query_history", [])[-window_size:]
    if history:
        history_text = " | ".join(history)
        return f"Previous questions: {history_text}. Current question: {current_question}"
    return current_question

#RAG chain with memory using LCEL
def create_rag_chain(vectorstore, embeddings):    
    hf_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task='conversational',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        temperature=0.3,
        max_new_tokens=512,
    )
    llm=ChatHuggingFace(llm=hf_llm)
    
    retriever = vectorstore.as_retriever(
        search_type='mmr', #Maximal marginal relevance for diverse retrieval
        search_kwargs={"k": 10,
                       "fetch-k": 20,
                       "lambda-mult": 0.5}
        )
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful, professional and a helpful AI assistant. The context may contain information from MULTIPLE documents.If a question asks about multiple topics, you MUST combine information from ALL relevant parts of the context.If something is missing, say exactly what is missing."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    
    base_chain = (
        {
            "context": (
            itemgetter("question") 
            | RunnableLambda(lambda q: build_retrieval_query(q, window_size=3))
            | retriever 
            | format_docs
            ),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    )

    return rag_chain, retriever

#UI
st.title("🤖 Personalized RAG Chat Assistant")
st.markdown("Upload documents and ask questions!")

with st.expander("ℹ️ API Configuration Status", expanded=False):
    if PINECONE_API_KEY != "YOUR_PINECONE_API_KEY_HERE":
        st.success("✅ Pinecone API Key: Configured")
    else:
        st.error("❌ Pinecone API Key: Not configured")
    if HUGGINGFACE_API_TOKEN != "YOUR_HUGGINGFACE_TOKEN_HERE":
        st.success("✅ HuggingFace Token: Configured")
    else:
        st.error("❌ HuggingFace Token: Not configured")
    

uploaded_files = st.file_uploader(
    "Upload a document (PDF, TXT, MD)",
    type=['pdf', 'txt', 'md'],
    help="Upload a document to build the knowledge base",
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = get_file_id(uploaded_file)
        if file_id not in st.session_state.uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    documents = load_document(uploaded_file)
                    chunks = chunk_documents(documents)
                    
                    embeddings = get_embeddings()
                    init_pinecone()
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = PineconeVectorStore.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            index_name=PINECONE_INDEX_NAME,
                            namespace=st.session_state.pinecone_namespace,
                            pinecone_api_key=PINECONE_API_KEY
                        )
                    else:
                        st.session_state.vectorstore.add_documents(chunks)

                    st.session_state.uploaded_files.add(file_id)
                    st.success(f"✅ Added {uploaded_file.name}! {len(chunks)} chunks created.")

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        else:
            st.info("Document already loaded. Ask questions below.")


st.markdown("---")
user_input = st.text_area(
    "Enter your question or text:",
    height=100,
    placeholder="Type your question here..."
)

if st.button("💬 Chat", type="primary", use_container_width=True):
    if not user_input:
        st.warning("Please enter a question or text.")
    elif not st.session_state.vectorstore:
        st.warning("Please upload a document first to build the knowledge base.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Generating response..."):
            try:
                embeddings = get_embeddings()
                rag_chain, retriever = create_rag_chain(st.session_state.vectorstore, embeddings)
                
                retrieved_docs = retriever.invoke(user_input)

                answer = rag_chain.invoke(
                    {"question": user_input},
                    config={"configurable": {"session_id": "default"}}
                    )
                
                st.session_state.user_query_history.append(user_input)
                st.session_state.user_query_history = st.session_state.user_query_history[-5:]

                metrics = calculate_retrieval_accuracy(user_input, retrieved_docs)
                st.session_state.retrieval_scores.append(metrics)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metrics": metrics
                })
                
            except Exception as e:
                import traceback
                st.error(f"Error generating response")
                st.code(traceback.format_exc())

st.markdown("---")
st.subheader("Conversation History")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            
            if "metrics" in message:
                metrics = message["metrics"]
                st.caption(
                    f"📊 Retrieval Metrics: {metrics['num_docs']} docs | "
                    f"Relevance: {metrics['relevance']} ({metrics['avg_score']}) | "
                    f"Time: {metrics['timestamp']}"
                )

if st.session_state.retrieval_scores:
    st.markdown("---")
    st.subheader("📊 Retrieval Performance Summary")
    
    avg_docs = sum(m["num_docs"] for m in st.session_state.retrieval_scores) / len(st.session_state.retrieval_scores)
    avg_relevance = sum(m["avg_score"] for m in st.session_state.retrieval_scores) / len(st.session_state.retrieval_scores)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", len(st.session_state.retrieval_scores))
    with col2:
        st.metric("Avg Documents Retrieved", f"{avg_docs:.1f}")
    with col3:
        st.metric("Avg Relevance Score", f"{avg_relevance:.3f}")

if st.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.session_state.retrieval_scores = []
    st.session_state.chat_history = ChatMessageHistory()
    st.rerun()

