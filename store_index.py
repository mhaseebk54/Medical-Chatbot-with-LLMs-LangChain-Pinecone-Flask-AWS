from dotenv import load_dotenv
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

# Load & split docs
extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Create index if not exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(5)

# Store documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("✅ Documents successfully indexed in Pinecone!")