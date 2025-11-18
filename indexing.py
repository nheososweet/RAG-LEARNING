

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

# 1.3: Embedding Generation and Vector Store Creation
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="kimetsu_no_yaiba",
    embedding_function=embedding_function,
    persist_directory="./chroma_db",
)

# vector_store.add_documents(chunks)

query = "Inosuke dùng hơi thở gì?"
print(f"❓ Câu hỏi: {query}\n")

# 3. TÌM KIẾM (Similarity Search)
# k=3 nghĩa là lấy 3 đoạn liên quan nhất
results = vector_store.similarity_search(query, k=3)

# 4. HIỂN THỊ KẾT QUẢ
print("--- 🔎 KẾT QUẢ TÌM KIẾM ---")
for i, doc in enumerate(results):
    print(f"\n[Kết quả #{i+1}]")
    print(f"Nội dung: {doc.page_content}")
    print(f"Nguồn: {doc.metadata}") # Xem nó đến từ file nào