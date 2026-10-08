from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def test_sql_generation():
    # 1. Connect to PostgreSQL
    # Replace YOUR_ACTUAL_PASSWORD with your pgAdmin master password
    db_uri = "postgresql+psycopg2://postgres:postgres@localhost:5432/data_narrator"
    db = SQLDatabase.from_uri(db_uri)
    schema = db.get_table_info()
    
    # 2. Initialize Local Llama 3.1
    # Setting temperature to 0 forces the AI to be analytical rather than creative
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    # 3. Create the Strict SQL Prompt
    template = """
    You are a PostgreSQL data engineer. Given the database schema below, write a SQL query that answers the user's question.
    CRITICAL: Output ONLY the raw SQL query. Do not include markdown formatting (like ```sql), conversational text, or explanations.

    Schema:
    {schema}

    Question: {question}
    SQL Query:
    """
    prompt = PromptTemplate.from_template(template)
    
    # 4. Build the LangChain Pipeline (Chain)
    chain = prompt | llm | StrOutputParser()
    
    # 5. Execute the test
    question = "How many products do we have in the Software category?"
    print(f"User Question: {question}\n")
    print("🧠 Llama 3.1 is analyzing the schema and writing SQL...\n")
    
    # Pass the schema and question into the chain
    response_sql = chain.invoke({
        "schema": schema,
        "question": question
    })
    
    # Clean up whitespace/markdown if the LLM ignores instructions
    response_sql = response_sql.strip().replace("```sql", "").replace("```", "")
    
    print("==================================================")
    print("✅ Generated SQL Query:")
    print(response_sql)
    print("==================================================")
    
    # 6. Execute the AI-generated SQL query against the real database
    try:
        query_result = db.run(response_sql)
        print(f"📊 Database Result: {query_result}")
    except Exception as e:
        print(f"❌ SQL Execution Failed: {e}")

if __name__ == "__main__":
    test_sql_generation()