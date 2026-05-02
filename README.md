# 🤖 Debales AI - Intelligent Support Agent (LangGraph + Streamlit)

A production-ready, agentic workflow built to act as "Alex," a friendly Customer Success Manager for Debales AI. This project uses LangGraph to intelligently route user queries between a local RAG knowledge base and a real-time web search API, all wrapped in a clean Streamlit web interface.

## ✨ Key Upgrades & Features

* **Human-Centric Web UI:** Upgraded from a CLI tool to a polished Streamlit interface that hides backend logs and provides a smooth conversational experience.
* **Persona-Driven Generation:** The agent adopts a professional, helpful persona ("Alex"), answering naturally without using robotic phrases like "According to the context..."
* **Conversational Intent Routing:** The LangGraph router doesn't just look for keywords; it understands pronouns and conversational context (e.g., "What services do *you* offer?" correctly routes to the company's internal RAG database).
* **Strict Anti-Hallucination:** If the context does not contain the answer, the agent gracefully declines and offers to connect the user with human support rather than hallucinating facts.

## 🧠 Architecture & Routing Logic

The core "brain" is a LangGraph `StateGraph` powered by the **Llama-3.1-8b-instant** model via Groq. The router node analyzes the user's query and directs it down one of four conditional paths:

1. **RAG Route (`rag_node`)**: For questions about Debales AI, it queries a local ChromaDB built from scraped company text (including Home, About, Blog, Contact, and Support pages).
2. **SERP Route (`serp_node`)**: For general knowledge and non-Debales queries, it searches the live web using the Tavily API.
3. **Mixed Route (Parallel Execution)**: For complex/comparative queries, LangGraph executes both the RAG and SERP tools simultaneously.
4. **Unknown Route (Fallback)**: Gibberish or unanswerable queries bypass the tools entirely and go straight to the generator to prevent API waste.

## 🛠️ Tech Stack
* **UI Framework:** Streamlit
* **Orchestration:** LangGraph & LangChain
* **LLM:** Groq (Llama-3.1-8b-instant)
* **Web Search:** Tavily API
* **Vector Database:** Chroma
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Scraping:** BeautifulSoup4

## 📁 Project Structure
* `app.py`: The frontend Streamlit UI and application entry point.
* `main.py`: The backend LangGraph state definition, routing logic, and system prompts.
* `knowledge_base.py`: The scraper script that builds the local ChromaDB vector store.
* `rag_tool.py`: Handles connection and similarity search against the local vector store.
* `serp_tool.py`: Handles web search requests to the Tavily API.

## 🚀 Setup & Installation Instructions

**1. Clone and set up the environment**
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install dependencies
```bash
pip install -r requirements.txt
```
**2. Configure API Keys**
Rename the .env.example file to .env and insert your API keys:

```Plaintext
GROQ_API_KEY="your_groq_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

**3. Build the Knowledge Base**
Run the scraper to initialize your local vector database:

```Bash
python knowledge_base.py
```

**4. Launch the Web App**
Start the Streamlit interface:

```Bash
streamlit run app.py
```

## 🧪 Evaluation Prompts to Test
Once the Web UI is running, try these exact prompts to see the LangGraph router and persona in action:

- Test 1 (RAG Route + Persona): "What services do you offer?"

- Test 2 (RAG Route - Contact Info): "How can I contact your team?"

- Test 3 (SERP Route): "What is the capital of Japan?"

- Test 4 (Mixed/Parallel Route): "Compare Debales AI to OpenAI."

- Test 5 (Anti-Hallucination/Unknown): "asdfghjkl"