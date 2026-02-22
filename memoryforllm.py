from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from tying import Tying
from langchain.schema import Document
from dotenv import load_dotenv
import os
load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


from pinecone import Pinecone 
pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

def load_data(document):
    loader= DirectoryLoader(document,glob="*.pdf",loader_cls=PyPDFLoader)
    data = loader.load()
    return data

extracted_data = load_data("data")



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

minimal_docs = filter_to_minimal_docs(extracted_data)

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

texts_chunk = text_split(minimal_docs)

from langchain.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

embedding = download_embeddings()



from pinecone import ServerlessSpec 

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)