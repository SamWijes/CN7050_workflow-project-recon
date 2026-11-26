import os
import json
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

DB_PATH = "./telecom_vectordb"
COLLECTION_NAME = "telecom_policies"
LOCAL_MODEL = "qwen2.5:7b"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
# llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

SOURCES = {
    "fault_manual": "telecom_fault_manual.txt",
    "contract": "contract_pricing_sla.txt",
}


def retrieve_policies(query: str, source: str = None, top_k: int = 2) -> list:
    """Retrieval only. The calling agent decides when to search and reasons over the result itself."""
    params = {"query_embeddings": embed_model.encode([query]).tolist(), "n_results": top_k}
    if source in SOURCES:
        params["where"] = {"source": SOURCES[source]}
    return collection.query(**params)["documents"][0]

def query_telecom_rag(user_query: str, source_filter: str = None, top_k: int = 2) -> dict:
    
    query_embedding = embed_model.encode([user_query]).tolist()
    
    query_params = {
        "query_embeddings": query_embedding,
        "n_results": top_k
    }
    if source_filter:
        query_params["where"] = {"source": source_filter}

    results = collection.query(**query_params)
    retrieved_docs = results["documents"][0]
    context = "\n---\n".join(retrieved_docs)

    # 2. Generate grounded JSON decision
    system_prompt = (
        "You are a Telecom Operations AI. Answer strictly using the provided policy context. "
        "Return valid JSON with keys: 'policy_code', 'summary', 'action_required', 'trigger_escalation_or_replenishment' (boolean)."
    )

    response = llm_client.chat.completions.create(
        # model="gpt-4o-mini",
        model="qwen2.5:7b",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Policy Context:\n{context}\n\nQuery/Log: {user_query}"}
        ]
    )

    return {
        "query": user_query,
        "retrieved_context": retrieved_docs,
        "agent_decision": json.loads(response.choices[0].message.content)
    }

if __name__ == "__main__":
    
    test_log = "Site #104: 5G Radio Unit SN-99812 failed to power on after tower mounting."
    output = query_telecom_rag(test_log, source_filter="telecom_fault_manual.txt")
    print(json.dumps(output, indent=2))