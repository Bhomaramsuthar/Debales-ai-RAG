import os
from typing import List, TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str         
    route: str           

class RouteDecision(BaseModel):
    decision: Literal["rag", "serp", "both", "unknown"] = Field(
        description="Route to 'rag' for Debales AI info, 'serp' for general info, 'both' if both are needed, or 'unknown' if it's gibberish."
    )

# Initialize Groq using Llama 3 8B (fast and smart enough for routing)
llm = ChatGroq(model="llama3-8b-8192", temperature=0)
router_llm = llm.with_structured_output(RouteDecision)

def route_query(state: AgentState):
    user_query = state["messages"][-1].content
    
    system_prompt = f"""
    You are a routing assistant for Debales AI. Analyze the user query.
    - If the query is about 'Debales', 'Debales AI', their products, or blogs: output 'rag'.
    - If the query is a general question (e.g., weather, history, coding help not related to Debales): output 'serp'.
    - If the query asks to compare Debales AI to something else on the web: output 'both'.
    - If the query is total gibberish or unanswerable: output 'unknown'.
    
    User Query: {user_query}
    """
    
    result = router_llm.invoke(system_prompt)
    return {"route": result.decision}