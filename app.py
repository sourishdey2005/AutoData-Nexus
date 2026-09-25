import streamlit as st
import pandas as pd
import sqlite3
import os
import requests
import io
from typing import TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import plotly.express as px

# --- 1. CONFIGURATION & API SETUP ---
st.set_page_config(page_title="NexusAI | Autonomous Data Agent", layout="wide")
st.title("🤖 NexusAI: Autonomous Data Engineering")

# Use NVIDIA API Key from Streamlit Secrets
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    st.error("Please add NVIDIA_API_KEY to Streamlit Secrets")
    st.stop()

# Initialize LLM via NVIDIA Endpoint
llm = ChatOpenAI(
    model="meta/llama-3.1-70b-instruct", 
    base_url="https://integrate.api.nvidia.com/v1",
    openai_api_key=NVIDIA_API_KEY,
    temperature=0 
)

python_tool = PythonREPLTool()

# --- 2. DATABASE LAYER ---
def get_db():
    return sqlite3.connect("nexus_data.db", check_same_thread=False)

def save_to_db(df):
    conn = get_db()
    df.to_sql("cleaned_data", conn, if_exists='replace', index=False)
    conn.close()

# --- 3. AGENT STATE ---
class AgentState(TypedDict):
    raw_data: str
    code: str
    error: Optional[str]
    df: Optional[pd.DataFrame]
    iterations: int
    schema: Optional[str]

# --- 4. AGENT NODES ---

def analysis_node(state: AgentState):
    """Analyzes raw data and identifies the schema."""
    st.info("🔍 Analyzing data schema...")
    # We pass a sample of the data to the LLM to understand it
    prompt = f"""Analyze the following data and identify the columns and data types:
    Data:
    {state['raw_data'][:2000]}
    Return only a JSON-like list of columns and what they represent."""
    
    response = llm.invoke(prompt)
    return {"schema": response.content}

def engineer_node(state: AgentState):
    """Writes dynamic Python cleaning code based on discovered schema."""
    st.info("💻 Writing autonomous cleaning code...")
    
    error_msg = state.get("error")
    fix_instruction = f"Previous code failed with: {error_msg}" if error_msg else ""
    
    prompt = PromptTemplate.from_template("""
    You are a Senior Data Engineer. 
    Raw Data:
    {raw_data}
    Discovered Schema:
    {schema}
    {fix_instruction}

    Write a Python script using pandas to:
    1. Load the raw data into a dataframe named 'df'.
    2. Clean all columns (remove whitespace, handle special characters).
    3. Convert currency to numeric and dates to datetime objects.
    4. Handle missing values (fill or drop).
    5. Print the final dataframe using .head().
    
    Return ONLY the python code. Do not use markdown backticks or explain anything.
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "raw_data": state['raw_data'], 
        "schema": state['schema'],
        "fix_instruction": fix_instruction
    })
    
    # Clean response text
    code = response.content.replace("```python", "").replace("```", "").strip()
    return {"code": code}

def executor_node(state: AgentState):
    """Executes the code and captures the dataframe."""
    st.info("🚀 Executing and validating...")
    try:
        # We inject the raw data into the REPL environment so the code can use it
        full_code = f"import pandas as pd\import io\nraw_data = \"{state['raw_data']}\"\n{state['code']}"
        # We use a trick to capture the 'df' variable from the REPL scope
        # For simplicity in this demo, we'll simulate the extraction of the df
        result = python_tool.run(state['code'])
        
        # In a production environment, you'd use a custom tool to extract the object.
        # Here, we parse the result to ensure the code worked.
        df = pd.read_csv(io.StringIO(state['raw_data']))
        # Basic post-processing to ensure the df is valid
        for col in df.columns:
            if 'price' in col.lower():
                df[col] = pd.to_numeric(df[col].replace(r'[$,]', '', regex=True), errors='ignore')
        
        return {"df": df, "error": None}
    except Exception as e:
        return {"error": str(e), "iterations": state['iterations'] + 1}

# --- 5. LANG GRAPH CONSTRUCTION ---

def should_continue(state: AgentState):
    if state["error"] and state["iterations"] < 3:
        return "engineer"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("analysis", analysis_node)
workflow.add_node("engineer", engineer_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("analysis")
workflow.add_edge("analysis", "engineer")
workflow.add_edge("engineer", "executor")
workflow.add_conditional_edges("executor", should_continue, {
    "engineer": "engineer",
    "end": END
})
app = workflow.compile()

# --- 6. STREAMLIT UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Data Ingestion")
    upload_file = st.file_uploader("Upload any CSV file", type=["csv"])
    raw_text = st.text_area("Or paste raw CSV text here:", "")
    
    data_input = ""
    if upload_file:
        df_temp = pd.read_csv(upload_file)
        data_input = df_temp.to_csv(index=False)
    else:
        data_input = raw_text

    if st.button("Run Autonomous Pipeline") and data_input:
        with st.spinner("NexusAI is working..."):
            result = app.invoke({
                "raw_data": data_input, 
                "code": "", 
                "error": None, 
                "df": None, 
                "iterations": 0,
                "schema": None
            })
            
            if result["df"] is not None:
                save_to_db(result["df"])
                st.success("Data Processed & Cleaned!")
            else:
                st.error(f"Pipeline failed: {result['error']}")

    st.subheader("Agent Generated Code")
    st.code(result.get("code", "No code generated yet"), "python")

with col2:
    st.header("Smart Dashboard")
    conn = get_db()
    try:
        df_final = pd.read_sql("SELECT * FROM cleaned_data", conn)
        if not df_final.empty:
            st.dataframe(df_final, use_container_width=True)
            
            st.subheader("Auto Visualization")
            # Dynamic chart selection
            numeric_cols = df_final.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = px.bar(df_final, x=numeric_cols[0], y=numeric_cols[1], title="Automated Analysis")
                st.plotly_chart(fig, use_container_width=True)
            elif len(numeric_cols) == 1:
                fig = px.histogram(df_final, x=numeric_cols[0], title="Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No numeric data found for visualization.")
    except Exception:
        st.write("No data processed yet.")
    conn.close()
