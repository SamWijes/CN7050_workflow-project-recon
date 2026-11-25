import os
import chromadb
from sentence_transformers import SentenceTransformer

KB_DIR = "knowledge_base"
DB_PATH = "./telecom_vectordb"
COLLECTION_NAME = "telecom_policies"

def load_and_chunk_files(directory, chunk_size=300, overlap=50):
    chunks = []
    metadatas = []
    ids = []
    doc_counter = 0

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    metadatas.append({"source": filename})
                    ids.append(f"doc_{doc_counter}")
                    doc_counter += 1
                start += chunk_size - overlap

    return chunks, metadatas, ids

def build_vector_db():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Chunking documents from '{KB_DIR}/'...")
    chunks, metadatas, ids = load_and_chunk_files(KB_DIR)
    print(f"Generated {len(chunks)} chunks.")

    print(f"Indexing into ChromaDB at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embeddings = embed_model.encode(chunks).tolist()
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("RAG knowledge base successfully indexed!")

if __name__ == "__main__":
    build_vector_db()