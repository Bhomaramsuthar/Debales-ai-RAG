import os
import urllib3
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import bs4

# 1. Suppress the insecure request warnings so your terminal stays clean
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Identify your scraper so websites don't block it
os.environ["USER_AGENT"] = "Debales-AI-Intern-Project/1.0" 

def build_vector_db():
    print("Scraping Debales AI website (Upgrading Knowledge Base)...")
    
    urls = [
        "https://debales.ai/", # Home
        "https://debales.ai/logistics", #  solution/logistics
        "https://debales.ai/ecommerce", #  solution/ecommerce
        "https://debales.ai/integrations", #  integrations
        "https://debales.ai/ai-agent" #  ai-agent
        "https://debales.ai/book-demo" #  book demo
        "https://debales.ai/blog" #  blog
        "https://debales.ai/case-studies" #  case studies
        "https://debales.ai/#faq" #  faq
        "https://debales.ai/case-study/email-ai-agentq" #  email ai agent
        "https://debales.ai/case-study/whatsapp-ai-agent" #  whatsapp ai agent
        "https://debales.ai/case-study/voice-ai-agent" #  voice ai agent
        "https://debales.ai/case-study/warehouse-ai-agent" #  warehouse ai agent
        "https://debales.ai/case-studies/debales-ai-cuts-customer-support-requests-for-blossom-and-rhyme" # ai agent
        "https://debales.ai/case-studies/debales-ai-turns-conversations-into-conversions-for-hellogtk" #  ai agent


    ]
    
    # 2. Tell the loader to ignore the SSL Certificate mismatch
    loader = WebBaseLoader(
        web_paths=urls,
        requests_kwargs={"verify": False} 
    )
    docs = loader.load()
    
    # 3. Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Created {len(splits)} text chunks. Generating embeddings...")
    
    # 4. Create free HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 5. Store in the local Chroma database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    print("Vector database built successfully!")
    
    return vectorstore

if __name__ == "__main__":
    build_vector_db()