import streamlit as st
import pandas as pd
import plotly.express as px
import json
from sqlalchemy import create_engine
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import io

# 1. State Definition (Updated with dataframe and chart_fig)
class GraphState(TypedDict):
    question: str
    sql_query: str
    result: str
    error: str
    loop_count: int
    summary: str 
    dataframe: str # Store data for plotting
    chart_fig: Any # Store the Plotly figure

# 2. Database and LLM Initialization
# (Replace YOUR_ACTUAL_PASSWORD)
db_uri = "postgresql+psycopg2://postgres:postgres@localhost:5432/data_narrator"
db = SQLDatabase.from_uri(db_uri)
engine = create_engine(db_uri) # Added SQLAlchemy engine for Pandas
schema = db.get_table_info()
llm = ChatOllama(model="llama3.1", temperature=0)

# 3. LangGraph Nodes
def write_query(state: GraphState):
    error_context = ""
    if state.get("error"):
        error_context = f"Your previous query failed with this error: {state['error']}\nPlease rewrite the query to fix it. Ensure you are using the correct column names."
        
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
    sql = chain.invoke({"schema": schema, "error_context": error_context, "question": state["question"]})
    clean_sql = sql.strip().replace("```sql", "").replace("```", "")
    return {"sql_query": clean_sql, "loop_count": state.get("loop_count", 0) + 1}

def execute_query(state: GraphState):
    sql = state["sql_query"]
    try:
        # Using Pandas to execute SQL so we capture column names!
        df = pd.read_sql(sql, engine)
        res_str = str(df.to_dict(orient="records"))
        return {"result": res_str, "dataframe": df.to_json(orient="split"), "error": ""} 
    except Exception as e:
        return {"error": str(e), "result": "", "dataframe": ""}

# ==========================================
# 🚀 NEW AGENT 4: Data Visualization Node
# ==========================================
def generate_chart(state: GraphState):
    q = state["question"].lower()
    
    # Intent Detection: Check if user actually wants a chart
    chart_keywords = ["plot", "chart", "graph", "visualize", "show me"]
    if not any(word in q for word in chart_keywords) or not state.get("dataframe"):
        return {"chart_fig": None}
        
    try:
        # FIXED LINE: io.StringIO() ব্যবহার করে Pandas-এর File Error ফিক্স করা হলো
        df = pd.read_json(io.StringIO(state["dataframe"]), orient="split")
        
        if df.empty or len(df.columns) < 2:
            return {"chart_fig": None} 
        
        template = """
        You are a Data Visualization Assistant.
        The user asked: {question}
        The available data columns are: {columns}
        
        Output ONLY a valid JSON object with EXACTLY these keys: "chart_type", "x", "y".
        - "chart_type" must be one of: "bar", "pie", "line".
        - "x" must be the column name for the X-axis (usually categorical/names).
        - "y" must be the column name for the Y-axis (usually numerical/totals).
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        config_str = chain.invoke({"question": state["question"], "columns": list(df.columns)})
        
        # Regex দিয়ে শুধুমাত্র JSON অংশটুকু এক্সট্র্যাক্ট করা
        json_match = re.search(r'\{.*?\}', config_str, re.DOTALL)
        
        if json_match:
            config = json.loads(json_match.group(0))
        else:
            # Fallback Logic
            config = {
                "chart_type": "bar",
                "x": df.columns[0],
                "y": df.columns[1]
            }
        
        # Chart Generation
        fig = None
        if config["chart_type"] == "bar":
            fig = px.bar(df, x=config["x"], y=config["y"], title=state["question"], template="plotly_white")
        elif config["chart_type"] == "pie":
            fig = px.pie(df, names=config["x"], values=config["y"], title=state["question"])
        elif config["chart_type"] == "line":
            fig = px.line(df, x=config["x"], y=config["y"], title=state["question"], markers=True)
            
        return {"chart_fig": fig}
    except Exception as e:
        print(f"Chart Agent Error: {e}")
        return {"chart_fig": None}

def summarize_answer(state: GraphState):
    template = """
    You are a helpful data analyst. A user asked: {question}
    The database returned this raw data: {result}
    
    Write a short, conversational sentence answering the question. Do not mention SQL or tuples.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"question": state["question"], "result": state["result"]})
    return {"summary": summary.strip()}

def should_continue(state: GraphState):
    if state["loop_count"] >= 3:
        return "end"
    if state.get("error"):
        return "retry"
    return "generate_chart" # Success? Send to Agent 4!

# 4. Compile Workflow
workflow = StateGraph(GraphState)
workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_chart", generate_chart) # Register Node 4
workflow.add_node("summarize_answer", summarize_answer)

workflow.set_entry_point("write_query")
workflow.add_edge("write_query", "execute_query")
workflow.add_conditional_edges(
    "execute_query",
    should_continue,
    {"retry": "write_query", "generate_chart": "generate_chart", "end": END}
)
# Routing sequence: Chart -> Summary -> END
workflow.add_edge("generate_chart", "summarize_answer")
workflow.add_edge("summarize_answer", END)
app = workflow.compile()


# ==========================================
# 5. STREAMLIT CHAT INTERFACE
# ==========================================
st.set_page_config(page_title="Data Narrator", page_icon="🤖", layout="wide")
st.title("🤖 Data Narrator Copilot")
st.markdown("Your offline AI Copilot for PostgreSQL.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            with st.expander("View Generated SQL"):
                st.code(message["sql"], language="sql")
        # Display Plotly chart if it exists in history
        if message.get("chart_fig"):
            st.plotly_chart(message["chart_fig"], use_container_width=True)

if prompt := st.chat_input("Ask a question (Tip: include 'plot' or 'chart' for visuals)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing schema, generating SQL, and rendering charts..."):
            final_state = app.invoke({
                "question": prompt,
                "sql_query": "",
                "result": "",
                "error": "",
                "loop_count": 0,
                "summary": "",
                "dataframe": "",
                "chart_fig": None
            })
            
            answer = final_state.get("summary", "Sorry, I couldn't generate an answer.")
            sql_query = final_state.get("sql_query", "")
            chart_fig = final_state.get("chart_fig", None)
            
            st.markdown(answer)
            with st.expander("View Generated SQL"):
                st.code(sql_query, language="sql")
            
            # Render the chart in real-time
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sql": sql_query,
                "chart_fig": chart_fig
            })