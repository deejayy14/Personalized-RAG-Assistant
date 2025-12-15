🤖 RAG Chat Assistant (LangChain v1 + Pinecone + Hugging Face + Streamlit)

A production style Retrieval Augmented Generation(RAG) application that allows users to upload multiple documents and ask conversational questions related to the uploaded content.

Tech Stack :
- Frontend          :   Streamlit  
- LLM               :   Mistral-7B-Instruct (Hugging Face)  
- Embeddings        :   sentence-transformers (MiniLM)  
- Vector Database   :   Pinecone  
- Framework         :   LangChain (v1.x)  

Architecture Overview :
User -> Streamlit UI -> Document Loader -> Chunking
     -> Embeddings (Hugging Face)
     -> Pinecone Vector DB (session namespace)
     -> MMR Retriever (for diverse retrieval of chunks)
     -> Hugging Face LLM (Mistral)
     -> Answer + Metrics

Key Features :
- Multiple document upload (PDF, TXT, MD)
- Session based knowledge base (documents are session-specific)
- MMR based retrieval for cross-document reasoning
- Conversational memory
- Retrieval relevance metrics
- Hallucination-safe responses (answers only from context)

Design Decisions :
- Session based Pinecone namespaces prevent data leakage between users
- MMR retrieval ensures diverse chunks across documents
- Evaluation metrics provide retrieval transparency

Architecture Explanation :
1. User Interface (Streamlit)
   - User uploads documents and asks questions
   - Maintains session based state
2. Document Processing
   - Documents are loaded 
   - Split into overlapping chunks
3. Embedding Layer
   - Text chunks are converted into embeddings using Hugging Face models
4. Vector Storage
   - Embeddings stored in Pinecone
   - Session based namespace storage to avoid data leakage
5. Retriever
   - MMR based diverse and semantic retrieval
   - Top k relevant chunks are selected
6. LLM Generation
   - Mistral-7B-Instruct generates relevant responses
   - Uses retrieved context + recent conversation history
7. Evaluation
   - Retrieval relevance scoring displayed per answer

Author:
Dhananjay,
B.Tech CSE(AI),
Aspiring ML/LLM Engineer.


🔧 Setup Instructions:
 ```bash
git clone https://github.com/your-username/Personalized-RAG-Assistant.git
cd Personalized-RAG-Assistant
pip install -r requirements.txt
streamlit run app.py
[Create a .env file:
HUGGINGFACE_API_TOKEN=your_token_here
PINECONE_API_KEY=your_key_here]



