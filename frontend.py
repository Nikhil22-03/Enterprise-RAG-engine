import streamlit as st
import requests
import uuid

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Llama 3 RAG Expert", page_icon="🦙")
st.title("🦙 ML Infrastructure Expert")
st.markdown("Ask me anything about the Llama 3 paper, FlashAttention, or PyTorch 2.0!")

# ==========================================
# 2. SESSION MANAGEMENT (The Invisible Handshake)
# ==========================================
# Generate a unique ID for this browser tab
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"user-{uuid.uuid4().hex[:8]}"
    
# Store the UI's local chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 3. RENDER CHAT HISTORY
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            st.caption(f"📚 **Sources:** {', '.join(msg['citations'])}")

# ==========================================
# 4. CHAT INPUT & API CALL
# ==========================================
if prompt := st.chat_input("Ask a technical question..."):
    # Immediately display user's question in the UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show a loading spinner while waiting for FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking (Querying Vector DB)..."):
            try:
                # Send the request to your FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={
                        "session_id": st.session_state["session_id"],
                        "query": prompt
                    }
                )
                response.raise_for_status() # Check for 500 errors
                data = response.json()
                
                answer = data["answer"]
                citations = data.get("citations", [])
                
                # Display the AI's answer and citations
                st.markdown(answer)
                if citations:
                    st.caption(f"📚 **Sources:** {', '.join(citations)}")
                    
                # Save the AI's response to the UI history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations
                })
                
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to backend. Is your FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"🚨 API Error: {e}")