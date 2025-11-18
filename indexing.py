

from langchain_community.document_loaders import TextLoader

# 1.1: Document Ingestion
loader = TextLoader(file_path="inosuke.txt", encoding="utf8")
documents = loader.load()

# 1.2: Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)
for chunk in chunks:
    print(f"Chunk: {chunk.page_content}\n")

# 1.3: Vector Store Creation
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="kimetsu_no_yaiba",
    embedding_function="all-MiniLM-L6-v2",
    persist_directory="./chroma_db",
)

vector_store.add_documents(chunks)