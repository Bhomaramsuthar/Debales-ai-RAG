import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# Load API keys from .env
load_dotenv()

def search_web(query: str) -> str:
    """Searches the web for general queries using Tavily."""
    
    # Initialize the Tavily tool to return the top 3 results
    try:
        tool = TavilySearch(max_results=3)
        # Store the full API dictionary response
        response = tool.invoke({"query": query})
    except Exception as e:
        return f"Error accessing the web search API: {e}"
    
    # Extract just the list of website hits from the 'results' key
    results_list = response.get("results", []) if isinstance(response, dict) else response
    
    # Format the results into a readable string for our LLM Generator
    if not results_list:
        return "No relevant web search results found."
        
    context = ""
    # Now we safely loop through the actual list of dictionaries
    for res in results_list:
        if isinstance(res, dict):
            context += f"Source: {res.get('url', 'Unknown')}\n"
            context += f"Content: {res.get('content', 'No content')}\n\n---\n\n"
        
    return context

# A quick local test to ensure your API key works!
if __name__ == "__main__":
    test_query = "What is the current weather in Pune Maharashtra?"
    print(f"Testing SERP pipeline with query: '{test_query}'\n")
    print("Searching the web...\n")
    
    result = search_web(test_query)
    print(result)