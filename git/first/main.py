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

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Optional: Disable MPS memory cap on Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Load model and tokenizer
model_id = "gpt2"  # Example model, change to your specific model if needed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Ensure pad token exists
if tokenizer.pad_token is None:
    print("No pad_token found, adding one...")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Set pad_token_id in model config
model.config.pad_token_id = tokenizer.pad_token_id

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
model.to(device)

# Example pipeline usage (CPU/GPU auto-handling)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type != "cpu" else -1  # GPU:0 or CPU:-1
)

# Input text for generation
input_text = "Artificial Intelligence is transforming"

# Tokenization with explicit padding and truncation
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=32
).to(device)

# Generate text
output_sequences = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

# Decode and print result
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print("\nGenerated Text:", generated_text)
