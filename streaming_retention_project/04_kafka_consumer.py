import json
import joblib
import requests
import pandas as pd
import chromadb
from confluent_kafka import Consumer, KafkaError
from sklearn.cluster import MiniBatchKMeans
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# 1. Custom Embedder for Retrieval
class OllamaLocalEmbedder(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            response = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": "nomic-embed-text", "input": doc}
            )
            embeddings.append(response.json()["embeddings"][0])
        return embeddings

# 2. Local Llama 3 Generator 
def generate_retention_offer(customer_profile, policy_text):
    prompt = f"""
    You are an AI retention agent for LTM Telecom.
    A high-risk customer is about to churn.
    
    Customer Profile: {customer_profile}
    Company Policy: {policy_text}
    
    Based ONLY on the company policy above, write a short, 1-2 sentence retention offer to save this customer. 
    Address the customer directly. Do not offer anything outside the policy.
    """
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False 
        }
    )
    return response.json()["response"].strip()

def start_consumer():
    print("=" * 70)
    print("⚙️ STARTING FULL RAG-ENABLED STREAMING CONSUMER")
    print("=" * 70)

    # 3. Load ML Pipeline & Warm-start clustering
    pipeline = joblib.load("models/xgboost_pipeline.pkl")
    df_baseline = pd.read_csv("data/Telco-Customer-Churn-Cleaned.csv")
    baseline_features = df_baseline[['tenure', 'MonthlyCharges', 'TotalCharges']].values
    
    clustering = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=20)
    clustering.fit(baseline_features)
    cluster_labels = {0: "Budget / High Risk", 1: "Core Stable", 2: "High-Value Enterprise"}
    
    # 4. Connect to local ChromaDB
    print("[⏳] Connecting to local ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
    embedder = OllamaLocalEmbedder()
    collection = chroma_client.get_collection(name="telecom_policies", embedding_function=embedder)
    print("[✓] Knowledge Base connected.")

    # 5. Configure Kafka Consumer
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'retention-rag-group',
        'auto.offset.reset': 'latest' # Changed to latest so we only process new events
    }
    consumer = Consumer(conf)
    consumer.subscribe(['customer-telemetry'])
    print("[✓] Subscribed to Kafka. Awaiting live stream...\n")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None: continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF and "UNKNOWN_TOPIC_OR_PART" not in str(msg.error()):
                    pass # Silencing Kafka background noise for cleaner output
                continue

            event = json.loads(msg.value().decode('utf-8'))
            input_df = pd.DataFrame([event])
            
            actual_churn = input_df.pop('Churn').values[0] if 'Churn' in input_df else 'N/A'
            if 'customerID' in input_df.columns:
                input_df.drop(columns=['customerID'], inplace=True)

            churn_prob = pipeline.predict_proba(input_df)[0][1]

            numeric_features = input_df[['tenure', 'MonthlyCharges', 'TotalCharges']].values
            clustering.partial_fit(numeric_features)
            assigned_cluster = clustering.predict(numeric_features)[0]

            status = "🚨 HIGH RISK" if churn_prob >= 0.70 else "✅ STABLE   "
            print(f"[{status}] P(Churn): {churn_prob:.3f} | Segment: {cluster_labels[assigned_cluster]:<22} | Actual: {actual_churn}")

            # --- THE RAG INTEGRATION TRIGGER ---
            if churn_prob >= 0.70:
                customer_profile = f"Tenure: {input_df['tenure'].values[0]} months, Monthly Charges: ${input_df['MonthlyCharges'].values[0]}, Contract: {input_df['Contract'].values[0]}"
                
                # Retrieve the matching policy
                results = collection.query(
                    query_texts=[customer_profile],
                    n_results=1
                )
                best_policy = results['documents'][0][0]
                
                clean_policy_print = best_policy.split('Action:')[0].strip().replace('\n', ' ')
                print(f"   ↳ 🔍 [CHROMA RETRIEVAL]: Found matching rule -> {clean_policy_print}")
                print(f"   ↳ 🧠 [LLM GENERATING]: Passing rule to Llama 3...")
                
                # Generate the offer
                offer = generate_retention_offer(customer_profile, best_policy)
                print(f"   ↳ 🎁 [RETENTION OFFER]: {offer}\n")

    except KeyboardInterrupt:
        print("\nStopping consumer service...")
    finally:
        consumer.close()

if __name__ == "__main__":
    start_consumer()