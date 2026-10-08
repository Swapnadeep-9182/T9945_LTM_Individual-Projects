from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Update State: আমরা এখানে 'summary' নামের নতুন একটি মেমরি ফিল্ড যোগ করেছি
class GraphState(TypedDict):
    question: str
    sql_query: str
    result: str
    error: str
    loop_count: int
    summary: str 

# Initialize global connections (Replace YOUR_ACTUAL_PASSWORD)
db_uri = "postgresql+psycopg2://postgres:postgres@localhost:5432/data_narrator"
db = SQLDatabase.from_uri(db_uri)
schema = db.get_table_info()
llm = ChatOllama(model="llama3.1", temperature=0)

# 2. Node 1: Generate SQL (আগের মতোই আছে)
def write_query(state: GraphState):
    print("\n[Node 1] 🧠 Generating SQL...")
    
    error_context = ""
    if state.get("error"):
        print(f"       ⚠️ Fixing previous error: {state['error'].split('LINE')[0].strip()}")
        error_context = f"Your previous query failed with this error: {state['error']}\nPlease rewrite the query to fix it. Ensure you are using the correct column names from the schema."
        
    template = """
    You are a PostgreSQL expert. Write a SQL query to answer the question based on the schema.
    Output ONLY the raw SQL query. No markdown, no explanations.

    Schema:
    {schema}
    
    {error_context}

    Question: {question}
    SQL Query:
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    sql = chain.invoke({
        "schema": schema, 
        "error_context": error_context, 
        "question": state["question"]
    })
    
    clean_sql = sql.strip().replace("```sql", "").replace("```", "")
    print(f"       ✅ Generated SQL: {clean_sql}")
    
    return {"sql_query": clean_sql, "loop_count": state.get("loop_count", 0) + 1}

# 3. Node 2: Execute SQL (আগের মতোই আছে)
def execute_query(state: GraphState):
    print("\n[Node 2] 🗄️ Executing SQL in PostgreSQL...")
    sql = state["sql_query"]
    
    try:
        res = db.run(sql)
        print(f"       ✅ Database Success! Raw Data: {res}")
        return {"result": res, "error": ""} 
    except Exception as e:
        error_msg = str(e)
        print(f"       ❌ Database Failed!")
        return {"error": error_msg, "result": ""}

# 4. NEW NODE 3: Summarize Answer (নতুন লজিক)
def summarize_answer(state: GraphState):
    print("\n[Node 3] ✍️ Summarizing Data into Natural Language...")
    
    template = """
    You are a helpful data analyst. 
    A user asked this question: {question}
    The database returned this raw data: {result}
    
    Please write a short, conversational sentence answering the user's question using the data provided. 
    Do not mention SQL, databases, or raw tuples in your response. Just give the final answer.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    # AI-কে ইউজার এর প্রশ্ন এবং ডেটাবেস এর রেজাল্ট পাঠিয়ে সুন্দর উত্তর তৈরি করতে বলছি
    summary = chain.invoke({
        "question": state["question"], 
        "result": state["result"]
    })
    
    print(f"       ✅ Summary Generated!")
    return {"summary": summary.strip()}

# 5. Conditional Edge: রাউটিং লজিক আপডেট করা হলো
def should_continue(state: GraphState):
    if state["loop_count"] >= 3:
        return "end"
    if state.get("error"):
        return "retry"
        
    # আগের মতো গ্রাফ শেষ না করে, ডেটাবেস সাকসেস হলে summarize নোডে পাঠানো হবে
    return "summarize"

def build_and_test_graph():
    # 6. Compile the Graph
    workflow = StateGraph(GraphState)
    
    workflow.add_node("write_query", write_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("summarize_answer", summarize_answer) # নতুন নোড রেজিস্টার করা হলো
    
    workflow.set_entry_point("write_query")
    workflow.add_edge("write_query", "execute_query")
    
    workflow.add_conditional_edges(
        "execute_query",
        should_continue,
        {
            "retry": "write_query",
            "summarize": "summarize_answer", # সাকসেস হলে Node 3 তে যাবে
            "end": END
        }
    )
    
    # Summarize নোড কাজ শেষ করার পর গ্রাফটি সম্পূর্ণ হবে
    workflow.add_edge("summarize_answer", END)
    
    app = workflow.compile()
    
    # 7. The Test
    test_question = "How many products do we have in the Software category?"
    print(f"\nUser Question: {test_question}")
    
    final_state = app.invoke({
        "question": test_question,
        "sql_query": "",
        "result": "",
        "error": "",
        "loop_count": 0,
        "summary": ""
    })
    
    print("\n" + "="*50)
    print("🎯 FINAL COPILOT RESPONSE:")
    if final_state.get("summary"):
        print(f"🤖 Data Narrator: {final_state['summary']}")
    else:
        print(f"Failed to resolve. Last error: {final_state.get('error')}")
    print("="*50)

if __name__ == "__main__":
    build_and_test_graph()