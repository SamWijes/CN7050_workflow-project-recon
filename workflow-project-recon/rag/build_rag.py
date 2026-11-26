import os
import chromadb
from sentence_transformers import SentenceTransformer

KB_DIR = "knowledge_base"
DB_PATH = "./telecom_vectordb"
COLLECTION_NAME = "telecom_policies"

def load_and_chunk_files(directory):
    """One chunk per policy entry (paragraphs separated by a blank line), so a rule is never split in half."""
    chunks, metadatas, ids = [], [], []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            text = f.read()
        for i, block in enumerate(p.strip() for p in text.split("\n\n")):
            if block:
                chunks.append(block)
                metadatas.append({"source": filename})
                ids.append(f"{filename}_{i}")
    return chunks, metadatas, ids

def build_vector_db():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Chunking documents from '{KB_DIR}/'...")
    chunks, metadatas, ids = load_and_chunk_files(KB_DIR)
    print(f"Generated {len(chunks)} chunks.")

    print(f"Indexing into ChromaDB at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)  # rebuild cleanly, no stale chunks
    except Exception:
        pass
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