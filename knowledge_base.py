import os
# Identify your scraper so websites don't block it
os.environ["USER_AGENT"] = "Debales-AI-Intern-Project/1.0"


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import bs4

def build_vector_db():
    print("Scraping Debales AI website...")
    
    # 1. Define the target URLs (Update these with the actual Debales URLs)
    urls = [
        "https://debales.ai/", # Example Home
        "https://debales.ai/logistics", # Example solution/logistics
        "https://debales.ai/ecommerce", # Example solution/ecommerce
        "https://debales.ai/integrations", # Example integrations
        "https://debales.ai/ai-agent" # Example ai-agent
        "https://debales.ai/book-demo" # Example book demo
        "https://debales.ai/blog" # Example blog
        "https://debales.ai/case-studies" # Example case studies
        "https://debales.ai/#faq" # Example faq
        "https://debales.ai/case-study/email-ai-agentq" # Example email ai agent
        "https://debales.ai/case-study/whatsapp-ai-agent" # Example whatsapp ai agent
        "https://debales.ai/case-study/voice-ai-agent" # Example voice ai agent
        "https://debales.ai/case-study/warehouse-ai-agent" # Example warehouse ai agent
        "https://debales.ai/case-studies/debales-ai-cuts-customer-support-requests-for-blossom-and-rhyme" # Example ai agent
        "https://debales.ai/case-studies/debales-ai-turns-conversations-into-conversions-for-hellogtk" # Example ai agent


    ]
    
    # 2. Load the web pages
    loader = WebBaseLoader(web_paths=urls)
    docs = loader.load()
    
    # 3. Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Created {len(splits)} text chunks. Generating embeddings...")
    
    # 4. Create free HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 5. Store in a local Chroma database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    print("Vector database built successfully!")
    
    return vectorstore

if __name__ == "__main__":
    build_vector_db()