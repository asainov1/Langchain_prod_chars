import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your CSV data
df = pd.read_csv("./2023-07-13-yc-companies.csv", usecols=['long_description'])
df = df.drop_duplicates()

# Convert rows to LangChain Document objects
docs = [Document(page_content=row['long_description']) for _, row in df.iterrows()]

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# ✅ Local Sentence Transformer Model (Free, No API Key Needed)
class LocalEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# Initialize local embedding model
local_embeddings = LocalEmbedding()

# Create vector store using local embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=local_embeddings)

# Prepare retriever
retriever = vectorstore.as_retriever()

# Define Prompt Template as a Python String
prompt_template = """
Use the following context to answer the user's question.
If you don't know the answer, just say you don't know.

Question: {question}
Context: {context}
Answer:
"""

# Simulate Retrieval + Prompt Execution
context_docs = retriever.get_relevant_documents("Generate a company idea for the HR industry")
context_text = "\n".join([doc.page_content for doc in context_docs])

# Prepare the final prompt by formatting the template with the retrieved context
# final_prompt = prompt_template.format(
#     question="Generate a company idea for the HR industry",
#     context=context_text
# )
#
