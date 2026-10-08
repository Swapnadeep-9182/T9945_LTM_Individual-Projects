import os
import chromadb
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# 1. Custom Local Embedder (Bypasses IT Firewalls)
class OllamaLocalEmbedder(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            # Using the modern /api/embed endpoint with a dedicated embedding model
            response = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": "nomic-embed-text", "input": doc}
            )
            data = response.json()
            
            # Catch errors (like a missing model) so it doesn't crash silently
            if "error" in data:
                print(f"\n[X] Ollama Error: {data['error']}")
                raise ValueError(f"Ollama Error: {data['error']}")
                
            # The new endpoint returns a list of arrays under the "embeddings" key
            embeddings.append(data["embeddings"][0])
        return embeddings

def build_retrieval_plane():
    print("="*50)
    print("🧠 PHASE 3: KNOWLEDGE BASE INDEXING (RAG)")
    print("="*50)
    
    # 2. Initialize the Local Embedded Database
    # This creates a folder right next to your script to save the vectors
    db_path = "./chroma_db_data"
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    embedder = OllamaLocalEmbedder()
    
    collection = chroma_client.get_or_create_collection(
        name="telecom_policies",
        embedding_function=embedder
    )
    
    # 3. Read the Raw Enterprise Data
    file_path = "knowledge_base/telecom_policies.txt"
    if not os.path.exists(file_path):
        print(f"[X] Policy file not found at {file_path}. Please create it first.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 4. Semantic Chunking Strategy
    # Splitting by "Policy ID:" so each policy becomes its own distinct vector
    chunks = content.split("Policy ID:")
    policies = ["Policy ID:" + chunk.strip() for chunk in chunks[1:] if chunk.strip()]

    print(f"[✓] Document parsed. Found {len(policies)} distinct policies.")

    # 5. Insert into Vector Database
    print("[⏳] Generating embeddings using local Llama 3 (This may take a moment)...")
    
    ids = [f"retention_rule_{i}" for i in range(len(policies))]
    
    # We "upsert" so you can run this script multiple times safely without duplicating data
    collection.upsert(
        documents=policies,
        ids=ids,
        metadatas=[{"source": "RET_SLA", "type": "discount_policy"} for _ in policies]
    )

    print(f"[✓] Success! Canonical knowledge embedded and saved to {db_path}")
    print("="*50)

if __name__ == "__main__":
    build_retrieval_plane()