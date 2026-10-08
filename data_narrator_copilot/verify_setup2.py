import sys
import platform
import socket
import requests

def verify_environment():
    print("="*50)
    print("🎓 CAPSTONE PROJECT ENVIRONMENT DIAGNOSTICS")
    print("Project: Data Narrator (Multi-Agent NL-to-SQL Copilot)")
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
    libraries = ['psycopg2', 'langchain', 'langgraph', 'streamlit']
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
        print("  [X] Ollama service is NOT running. (Make sure Ollama is running in the background)")

    # 4. Check PostgreSQL Engine
    print("\n🗄️ Checking PostgreSQL Database:")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 5432))
    if result == 0:
        print("  [✓] PostgreSQL is RUNNING and listening on port 5432.")
    else:
        print("  [X] PostgreSQL is NOT reachable on port 5432.")
    sock.close()
        
    print("\n" + "="*50)
    print("Diagnostics Complete. If all checks are [✓], the system is READY.")
    print("="*50)

if __name__ == "__main__":
    verify_environment()