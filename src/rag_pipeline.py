from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

base_dir = Path(__file__).resolve().parent.parent
chroma_db_path = "/chroma_db"
docs_dir = str(base_dir / "rag_docs/")
chunk_size_ = 1000
chunk_overlap_ = 200

def load_or_create_vector_store():
    vector_store_path = Path(chroma_db_path)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load existing store if present
    if vector_store_path.exists() and any(vector_store_path.iterdir()):
        print("Loading existing vector store...")
        return Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)

    # Otherwise, create new vector store
    print("Creating new vector store from documents...")
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_,
        chunk_overlap=chunk_overlap_,
        length_function=len,
        is_separator_regex = False
    )
    chunks = text_splitter.split_documents(documents)

    vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    vector_store.add_documents(chunks)
    return vector_store