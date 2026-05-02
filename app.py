import streamlit as st
from langchain_core.messages import HumanMessage

# Import your compiled LangGraph app from main.py
# (This is why having all the logic in main.py was a great idea!)
from main import app as agent_app

# --- 1. Page Configuration ---
st.set_page_config(page_title="Debales AI Support", page_icon="🤖", layout="centered")
st.title("👋 Debales AI Customer Support")
st.markdown("Chat with Alex, our Customer Success Manager. Ask about our services, integrations, or how we can help your business.")

# --- 2. Session State Initialization ---
# This keeps track of the chat history on the screen
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm Alex from Debales AI. How can I help you today?"}]

# --- 3. Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. Chat Input & Processing ---
if prompt := st.chat_input("Ask a question..."):
    
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show a loading spinner while the LangGraph router does its background work
    with st.chat_message("assistant"):
        with st.spinner("Checking our resources..."):
            
            try:
                # Format the input for LangGraph
                inputs = {"messages": [HumanMessage(content=prompt)], "context": ""}
                
                # Run the graph! 
                # (This triggers your router, tools, and generator invisibly)
                final_state = agent_app.invoke(inputs)
                
                # Extract the final answer
                response_text = final_state["messages"][-1].content
                
                # Display the answer
                st.markdown(response_text)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")