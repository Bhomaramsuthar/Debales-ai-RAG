import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import your custom tools
from rag_tool import retrieve_debales_info
from serp_tool import search_web

load_dotenv()

# 1. Define the State
# operator.add allows multiple nodes to append context at the same time (parallel execution)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: Annotated[str, operator.add] 
    route: str

# 2. Define the Router Schema
class RouteDecision(BaseModel):
    decision: Literal["rag", "serp", "both", "unknown"] = Field(
        description="Route to 'rag' for Debales AI info, 'serp' for general info, 'both' if both are needed, or 'unknown' if gibberish."
    )

# 3. Initialize the Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
router_llm = llm.with_structured_output(RouteDecision)

# --- NODE FUNCTIONS ---

def router_node(state: AgentState):
    """Analyzes the query and decides the route."""
    user_query = state["messages"][-1].content
    system_prompt = f"""
    You are a routing assistant for Debales AI. Analyze the user query.
    - If the query is about 'Debales', 'Debales AI', their products, or blogs: output 'rag'.
    - If the query is a general question (e.g., weather, history, coding help): output 'serp'.
    - If the query asks to compare Debales AI to something else on the web: output 'both'.
    - If the query is total gibberish or unanswerable: output 'unknown'.
    
    User Query: {user_query}
    """
    result = router_llm.invoke(system_prompt)
    return {"route": result.decision}

def rag_node(state: AgentState):
    """Fetches data from your local ChromaDB."""
    user_query = state["messages"][-1].content
    context = retrieve_debales_info(user_query)
    return {"context": f"\n\n[DEBALES AI KNOWLEDGE BASE]:\n{context}"}

def serp_node(state: AgentState):
    """Fetches data from the web using Tavily."""
    user_query = state["messages"][-1].content
    context = search_web(user_query)
    return {"context": f"\n\n[WEB SEARCH RESULTS]:\n{context}"}

def generator_node(state: AgentState):
    """Generates the final hallucination-free response."""
    user_query = state["messages"][-1].content
    context = state.get("context", "")
    route = state.get("route", "unknown")
    
    # Handle the "Unknown" edge case immediately to prevent hallucination
    if route == "unknown":
        return {"messages": [AIMessage(content="I'm sorry, I don't understand your question or it is out of my scope.")]}
    
    # DEBUG: This will print in your terminal so you know the data arrived!
    print(f"\n   [DEBUG] Context loaded into Generator: {len(context)} characters")
        
    system_prompt = f"""You are a helpful assistant for Debales AI. 
    Answer the user's query based ONLY on the provided Context below. 
    
    Rules:
    - If the user asks what Debales AI is, summarize the features, partnerships, or services mentioned in the Context to explain what they do.
    - If the context is completely empty or contains no relevant information at all, say "I don't have enough information to answer that." 
    - Do not guess or hallucinate outside of the Context.
    
    Context:
    {context}
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ])
    
    return {"messages": [response]}


# --- CONDITIONAL ROUTING LOGIC ---

def decide_next_node(state: AgentState):
    """Reads the router's decision and directs traffic."""
    route = state.get("route")
    if route == "rag":
        return "rag_node"
    elif route == "serp":
        return "serp_node"
    elif route == "both":
        # Returning a list executes both nodes in parallel
        return ["rag_node", "serp_node"] 
    else:
        return "generator_node"

# --- BUILD THE GRAPH ---

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("router_node", router_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("serp_node", serp_node)
workflow.add_node("generator_node", generator_node)

# Connect the edges
workflow.add_edge(START, "router_node")
workflow.add_conditional_edges("router_node", decide_next_node)
workflow.add_edge("rag_node", "generator_node")
workflow.add_edge("serp_node", "generator_node")
workflow.add_edge("generator_node", END)

# Compile!
app = workflow.compile()

# --- COMMAND LINE INTERFACE ---

if __name__ == "__main__":
    print("🤖 Debales AI Agent Initialized. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Initialize state with the user's message and empty context
        inputs = {"messages": [HumanMessage(content=user_input)], "context": ""}
        
        # Stream the graph execution so we can see the nodes firing
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"   ⚙️  Running: {key}...")
                
        # Get the final generated message
        final_message = value["messages"][-1].content
        print(f"\nDebales Assistant: {final_message}\n")
        print("-" * 50 + "\n")