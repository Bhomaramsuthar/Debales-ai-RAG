# 🤖 Debales AI - Agentic Workflow Assignment

A LangGraph-powered AI assistant that intelligently routes queries between a local RAG knowledge base and a real-time web search API to provide accurate, hallucination-free answers.

## 🧠 Architecture & Routing Logic

The core "brain" of this agent is a LangGraph `StateGraph` powered by the **Llama-3.1-8b-instant** model via Groq. The router node analyzes the user's query and directs it down one of four conditional paths to ensure optimal context retrieval:

1. **RAG Route (`rag_node`)**: For questions specifically about Debales AI, it queries a local ChromaDB. The database was built by scraping the Debales website and embedding the text using HuggingFace (`all-MiniLM-L6-v2`).
2. **SERP Route (`serp_node`)**: For general knowledge and non-Debales queries, it searches the live web using the Tavily API.
3. **Mixed Route (Parallel Execution)**: For complex or comparative queries (e.g., "Compare Debales AI to OpenAI"), LangGraph executes both the RAG and SERP tools simultaneously to gather comprehensive context.
4. **Unknown Route (Fallback)**: Gibberish or completely unanswerable queries bypass the tools entirely and go straight to the generator to prevent API waste and hallucinations.

The final `generator_node` synthesizes the retrieved context and operates under strict system prompts. If the context does not contain the answer, the agent is instructed to gracefully decline rather than hallucinate.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph & LangChain
* **LLM:** Groq (Llama-3.1-8b-instant)
* **Web Search:** Tavily API
* **Vector Database:** Chroma
* **Embeddings:** HuggingFace
* **Scraping:** BeautifulSoup4

## 📁 Project Structure
* `main.py`: The entry point. Contains the LangGraph state definition, routing logic, and the CLI loop.
* `knowledge_base.py`: The scraper script that builds the local ChromaDB vector store.
* `rag_tool.py`: Handles connection and similarity search against the local vector store.
* `serp_tool.py`: Handles web search requests to the Tavily API.

## 🚀 Setup & Installation Instructions

**1. Clone and set up the environment**
### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  
# On Windows use `venv\Scripts\activate`
```

### Install dependencies
```bash
pip install -r requirements.txt
```

**2. Configure API Keys
### Rename the .env.example file to .env and insert your API keys:
```Plaintext
GROQ_API_KEY="your_groq_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```
**3. Build the Knowledge Base
### Run the scraper to initialize your local vector database:

```bash
python knowledge_base.py
```
**4. Run the Agent
Start the interactive CLI:

```bash
python main.py
```
## 🧪 Evaluation Prompts to Test
Once the CLI is running, try these exact prompts to see the LangGraph router in action:

- Test 1 (RAG Route): "What is Debales AI?"

- Test 2 (SERP Route): "What is the capital of Japan?"

- Test 3 (Mixed/Parallel Route): "Compare Debales AI to OpenAI."

- Test 4 (Anti-Hallucination/Unknown): "asdfghjkl"