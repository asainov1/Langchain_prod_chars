import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your CSV data (replace with your file if needed)
df = pd.DataFrame({"text": ['Alikhan working in',"AI transforms industries.", "Python is great for AI.", "LangChain enables LLM apps."]})

# Convert rows to LangChain Document objects
docs = [Document(page_content=row["text"]) for _, row in df.iterrows()]

# Split documents into smaller chunks if needed
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
split_docs = splitter.split_documents(docs)

# Use Local Sentence Transformer Model (No API Key Needed)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert split_docs to plain text for embeddings
texts = [doc.page_content for doc in split_docs]
embeddings = embedding_model.encode(texts, show_progress_bar=True)

# Create Chroma vector store
vectorstore = Chroma.from_texts(texts, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

# Example query
query = "What is LangChain?"
docs_and_scores = vectorstore.similarity_search_with_score(query, k=2)

# Show results
for doc, score in docs_and_scores:
    print(f"\nFound: {doc.page_content} (Score: {score})")