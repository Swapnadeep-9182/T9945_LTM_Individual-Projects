from langchain_community.utilities import SQLDatabase

def test_langchain_connection():
    # The URI format is: dialect+driver://username:password@host:port/database
    # Replace YOUR_ACTUAL_PASSWORD with your pgAdmin master password
    db_uri = "postgresql+psycopg2://postgres:postgres@localhost:5432/data_narrator"
    
    try:
        # Initialize the LangChain database wrapper
        db = SQLDatabase.from_uri(db_uri)
        print("="*50)
        print("✅ LangChain successfully connected to PostgreSQL!")
        print("="*50)
        
        # Fetch the table names
        tables = db.get_usable_table_names()
        print(f"\n📊 Detected Tables: {tables}")
        
        # Fetch the schema of the products table to prove LangChain can read it
        print("\n📝 Schema mapping for the AI agent (Products Table):")
        print(db.get_table_info(["products"]))
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    test_langchain_connection()