import sys
import platform
import subprocess
import requests

def verify_environment():
    print("="*50)
    print("🎓 CAPSTONE PROJECT ENVIRONMENT DIAGNOSTICS")
    print("Project: Real-Time Streaming Customer Retention")
    print("="*50)
    
    # 1. Check Python Version and Environment
    print(f"\n[✓] OS Platform: {platform.system()} {platform.release()}")
    print(f"[✓] Python Version: {sys.version.split()[0]}")
    
    # Check if running inside a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("[✓] Virtual Environment: ACTIVE")
    else:
        print("[X] Virtual Environment: INACTIVE (Please activate venv!)")

    # 2. Check Critical Libraries
    libraries = ['pandas', 'xgboost', 'sklearn', 'langchain', 'chromadb', 'confluent_kafka']
    print("\n📦 Checking Core Libraries:")
    for lib in libraries:
        try:
            __import__(lib)
            print(f"  [✓] {lib} is installed.")
        except ImportError:
            print(f"  [X] {lib} is MISSING.")

    # 3. Check local Ollama execution
    print("\n🧠 Checking Offline LLM (Ollama):")
    try:
        response = requests.get("http://localhost:11434/")
        if response.status_code == 200:
            print("  [✓] Ollama background service is RUNNING on port 11434.")
        else:
            print("  [!] Ollama responded, but with an unexpected status code.")
    except requests.exceptions.ConnectionError:
        print("  [X] Ollama service is NOT running. (Make sure Ollama is open/running in the background)")
        
    print("\n" + "="*50)
    print("Diagnostics Complete. If all checks are [✓], the system is READY.")
    print("="*50)

if __name__ == "__main__":
    verify_environment()