import json
import joblib
import requests
import pandas as pd
import chromadb
import uuid
import asyncio
from confluent_kafka import Consumer
from sklearn.cluster import MiniBatchKMeans
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# --- 1. RAG & MODEL CLASSES ---
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

def generate_retention_offer(customer_profile, policy_text):
    prompt = f"""
    You are an AI retention agent for LTM Telecom. A high-risk customer is about to churn.
    Customer Profile: {customer_profile}
    Company Policy: {policy_text}
    Based ONLY on the policy, write a short, 1-2 sentence retention offer. Address the customer directly.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()

# --- 2. ENGINE INITIALIZATION ---
pipeline = joblib.load("models/xgboost_pipeline.pkl")
df_baseline = pd.read_csv("data/Telco-Customer-Churn-Cleaned.csv")
baseline_features = df_baseline[['tenure', 'MonthlyCharges', 'TotalCharges']].values
clustering = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=20)
clustering.fit(baseline_features)
cluster_labels = {0: "Budget / High Risk", 1: "Core Stable", 2: "High-Value Enterprise"}

chroma_client = chromadb.PersistentClient(path="./chroma_db_data")
embedder = OllamaLocalEmbedder()
collection = chroma_client.get_collection(name="telecom_policies", embedding_function=embedder)

app = FastAPI()

# --- 3. PURE HTML/CSS/JS FRONTEND ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LTM Retention Platform</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 30px; }
        h1, h2, h3 { color: #2c3e50; }
        .header { text-align: center; margin-bottom: 40px; border-bottom: 2px solid #ecf0f1; padding-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 4px solid #3498db; }
        .card h3 { margin: 0 0 10px 0; font-size: 13px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
        .card p { margin: 0; font-size: 28px; font-weight: bold; color: #2c3e50; }
        .status-stable { color: #27ae60 !important; }
        .status-risk { color: #e74c3c !important; }
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        
        /* Two-Column Log Layout */
        .logs-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .log-section-title { border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-bottom: 15px; }
        .log-container { background: white; padding: 0; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); max-height: 600px; overflow-y: auto; }
        .log-entry { padding: 15px 20px; border-bottom: 1px solid #ecf0f1; display: flex; flex-direction: column; gap: 8px; }
        
        .log-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        .badge-stable { background: #e8f8f5; color: #27ae60; border: 1px solid #27ae60; }
        .badge-risk { background: #fdedec; color: #c0392b; border: 1px solid #c0392b; }
        .log-rule { background: #fff8e1; color: #856404; padding: 10px; border-radius: 4px; font-size: 13px; border-left: 4px solid #ffc107; margin-top: 5px; }
        .log-offer { background: #e8f5e9; color: #1b5e20; padding: 10px; border-radius: 4px; font-size: 14px; border-left: 4px solid #4caf50; margin-top: 5px; font-weight: 500; }
        .loading-offer { background: #e3f2fd; color: #1565c0; padding: 10px; border-radius: 4px; font-size: 13px; border-left: 4px solid #1976d2; margin-top: 5px; font-style: italic; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📡 Live Streaming Retention Platform</h1>
        <p style="color: #7f8c8d;">Monitoring enterprise telemetry via Apache Kafka & Llama 3 (Asynchronous MLOps Edition)</p>
    </div>

    <h2>📊 Live Session Analytics</h2>
    <div class="grid">
        <div class="card"><h3>Total Customers Scanned</h3><p id="tot-scanned">0</p></div>
        <div class="card"><h3>Total At-Risk</h3><p id="tot-risk">0</p></div>
        <div class="card"><h3>Revenue at Risk (Monthly)</h3><p id="tot-rev" class="status-risk">$0.00</p></div>
        <div class="card"><h3>Latest System Status</h3><p id="latest-status" class="status-stable">WAITING...</p></div>
    </div>

    <div class="charts">
        <div class="chart-container"><canvas id="barChart"></canvas></div>
        <div class="chart-container"><canvas id="scatterChart"></canvas></div>
    </div>

    <!-- TWO COLUMN LOGS -->
    <div class="logs-layout">
        <!-- Left Column: Normal Log -->
        <div>
            <h2 class="log-section-title">📝 Live Event Log (All Data)</h2>
            <div class="log-container" id="all-log"></div>
        </div>
        
        <!-- Right Column: Risky Customers Only -->
        <div>
            <h2 class="log-section-title">🚨 Actionable Risk Queue (Latest 15)</h2>
            <div class="log-container" id="risk-log" style="border: 2px solid #fdedec;"></div>
        </div>
    </div>

    <script>
        const ctxBar = document.getElementById('barChart').getContext('2d');
        const barChart = new Chart(ctxBar, { type: 'bar', data: { labels: ['Month-to-month', 'One year', 'Two year'], datasets: [{ label: 'At-Risk Count by Contract', data: [0, 0, 0], backgroundColor: '#3498db' }] }, options: { responsive: true, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } } });

        const ctxScatter = document.getElementById('scatterChart').getContext('2d');
        const scatterChart = new Chart(ctxScatter, { type: 'scatter', data: { datasets: [] }, options: { responsive: true, animation: false, plugins: { title: { display: true, text: 'Cluster Distribution (Tenure vs Monthly Charges)' } }, scales: { x: { title: { display: true, text: 'Tenure (Months)' } }, y: { title: { display: true, text: 'Monthly Charges ($)' } } } } });

        let totalScanned = 0, totalRisk = 0, revenueRisk = 0.0;
        let contractRisk = { 'Month-to-month': 0, 'One year': 0, 'Two year': 0 };
        let scatterData = { 'Budget / High Risk': [], 'Core Stable': [], 'High-Value Enterprise': [] };
        const colors = { 'Budget / High Risk': '#e74c3c', 'Core Stable': '#2ecc71', 'High-Value Enterprise': '#f39c12' };

        const source = new EventSource("/stream");
        source.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === "metrics") {
                totalScanned++;
                if (data.is_high_risk) {
                    totalRisk++;
                    revenueRisk += data.monthly_charges;
                    contractRisk[data.contract]++;
                }

                document.getElementById('tot-scanned').innerText = totalScanned;
                document.getElementById('tot-risk').innerText = totalRisk;
                document.getElementById('tot-rev').innerText = "$" + revenueRisk.toFixed(2);
                document.getElementById('latest-status').innerText = data.is_high_risk ? "🚨 HIGH RISK DETECTED" : "✅ STREAMING SECURELY";
                document.getElementById('latest-status').className = data.is_high_risk ? "status-risk" : "status-stable";

                barChart.data.datasets[0].data = [contractRisk['Month-to-month'], contractRisk['One year'], contractRisk['Two year']];
                barChart.update();

                scatterData[data.segment].push({ x: data.tenure, y: data.monthly_charges });
                if(scatterData[data.segment].length > 30) scatterData[data.segment].shift(); 
                scatterChart.data.datasets = Object.keys(scatterData).map(key => ({ label: key, data: scatterData[key], backgroundColor: colors[key] }));
                scatterChart.update();

                // LEFT COLUMN: All Logs
                const allLogDiv = document.getElementById('all-log');
                const allEntry = document.createElement('div');
                allEntry.className = 'log-entry';
                allEntry.innerHTML = `<div><span class="log-badge ${data.is_high_risk ? 'badge-risk' : 'badge-stable'}">${data.is_high_risk ? '🚨 HIGH RISK' : '✅ STABLE'}</span> 
                               <span style="color: #7f8c8d; font-size: 13px; margin-left: 10px;">
                               <strong>P(Churn):</strong> ${data.churn_prob.toFixed(3)} | <strong>Segment:</strong> ${data.segment}
                               </span></div>`;
                allLogDiv.prepend(allEntry);
                if (allLogDiv.children.length > 25) allLogDiv.removeChild(allLogDiv.lastChild); 

                // RIGHT COLUMN: Only Risky Logs with Offer Placeholders
                if (data.is_high_risk) {
                    const riskLogDiv = document.getElementById('risk-log');
                    const riskEntry = document.createElement('div');
                    riskEntry.className = 'log-entry';
                    riskEntry.innerHTML = `<div><span class="log-badge badge-risk">🚨 P(Churn): ${data.churn_prob.toFixed(3)}</span> 
                                   <span style="color: #7f8c8d; font-size: 13px; margin-left: 10px;">
                                   <strong>Segment:</strong> ${data.segment} | <strong>Actual:</strong> ${data.actual_churn}
                                   </span></div>
                                   <div class="log-rule"><strong>🔍 Policy:</strong> ${data.rule}</div>
                                   <div class="loading-offer" id="offer-${data.event_id}"><em>⏳ AI Agent is analyzing policy and drafting offer...</em></div>`;
                    
                    riskLogDiv.prepend(riskEntry);
                    // Updated limit to 15
                    if (riskLogDiv.children.length > 15) riskLogDiv.removeChild(riskLogDiv.lastChild); 
                }
            } 
            
            else if (data.type === "offer") {
                const offerDiv = document.getElementById(`offer-${data.event_id}`);
                if (offerDiv) {
                    offerDiv.className = "log-offer";
                    offerDiv.innerHTML = `<strong>🤖 Llama 3 Retention Offer:</strong><br>${data.offer}`;
                }
            }
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get_dashboard():
    return HTMLResponse(HTML_TEMPLATE)

@app.get("/stream")
async def sse_stream():
    queue = asyncio.Queue()
    bg_tasks = set()

    async def kafka_reader():
        conf = {
            'bootstrap.servers': 'localhost:9092',
            'group.id': f'fastapi-dashboard-{uuid.uuid4()}',
            'auto.offset.reset': 'latest'
        }
        consumer = Consumer(conf)
        consumer.subscribe(['customer-telemetry'])
        
        try:
            while True:
                msg = await asyncio.to_thread(consumer.poll, 1.0)
                if msg is None or msg.error():
                    await asyncio.sleep(0.1)
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
                
                segment_name = cluster_labels[assigned_cluster]
                monthly_charges = float(input_df['MonthlyCharges'].values[0])
                tenure = float(input_df['tenure'].values[0])
                contract_type = input_df['Contract'].values[0]
                is_high_risk = bool(churn_prob >= 0.70)
                
                event_id = str(uuid.uuid4())
                
                payload = {
                    "type": "metrics",
                    "event_id": event_id,
                    "churn_prob": float(churn_prob),
                    "segment": segment_name,
                    "monthly_charges": monthly_charges,
                    "tenure": tenure,
                    "contract": contract_type,
                    "actual_churn": actual_churn,
                    "is_high_risk": is_high_risk,
                    "rule": ""
                }
                
                if is_high_risk:
                    customer_profile = f"Tenure: {tenure} months, Monthly Charges: ${monthly_charges}, Contract: {contract_type}"
                    results = collection.query(query_texts=[customer_profile], n_results=1)
                    best_policy = results['documents'][0][0]
                    clean_rule = best_policy.split('Action:')[0].strip().replace('\n', ' ')
                    payload["rule"] = clean_rule
                    
                    async def fetch_llm(profile, rule, eid):
                        offer = await asyncio.to_thread(generate_retention_offer, profile, rule)
                        await queue.put({"type": "offer", "event_id": eid, "offer": offer})
                    
                    task = asyncio.create_task(fetch_llm(customer_profile, best_policy, event_id))
                    bg_tasks.add(task)
                    task.add_done_callback(bg_tasks.discard)
                
                await queue.put(payload)
                
        finally:
            consumer.close()

    reader_task = asyncio.create_task(kafka_reader())
    bg_tasks.add(reader_task)
    reader_task.add_done_callback(bg_tasks.discard)

    async def event_generator():
        while True:
            data = await queue.get()
            yield f"data: {json.dumps(data)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8501)