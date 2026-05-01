from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def retrieve_debales_info(query: str) -> str:
    """Retrieves relevant context from the local Chroma vector database."""
    
    # 1. Initialize the EXACT same embedding model you used to build the DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to the existing local database folder
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 3. Perform a similarity search to find the top 3 most relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    
    # 4. Combine the text from the results into a single string for the LLM
    if not results:
        return "No relevant information found in the Debales AI knowledge base."
        
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    return context

# A quick local test to ensure your retrieval works!
if __name__ == "__main__":
    test_query = "What is Debales AI?"
    print(f"Testing RAG pipeline with query: '{test_query}'\n")
    print("Retrieving context...\n")
    
    result = retrieve_debales_info(test_query)
    print(result)