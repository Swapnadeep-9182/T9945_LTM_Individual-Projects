import json
import time
import pandas as pd
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f"[X] Delivery failed: {err}")

def start_producer():
    print("=" * 50)
    print("📡 STARTING KAFKA STREAM PRODUCER")
    print("=" * 50)

    conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(conf)
    topic = "customer-telemetry"

    df = pd.read_csv("data/Telco-Customer-Churn-Cleaned.csv")
    print(f"[✓] Streaming {len(df)} records to topic '{topic}'...\n")

    for idx, row in df.iterrows():
        payload = row.to_dict()
        producer.produce(
            topic,
            key=str(payload.get('customerID', idx)),
            value=json.dumps(payload),
            callback=delivery_report
        )
        producer.poll(0)
        print(f"-> Emitted Event #{idx + 1} | MonthlyCharges: ${payload['MonthlyCharges']} | Contract: {payload['Contract']}")
        time.sleep(6.0)

    producer.flush()

if __name__ == "__main__":
    start_producer()