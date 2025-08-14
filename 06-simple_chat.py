# simple_chat.py - Run this for instant chat interface
import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chat",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    padding: 1rem 0;
}
.crisis-warning {
    background-color: #ffebee;
    border: 1px solid #f44336;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    color: #d32f2f;
}
.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
}
.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}
.assistant-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 Mental Health Support Chatbot</h1>', unsafe_allow_html=True)
st.markdown("*Get support based on real community experiences from Reddit mental health communities*")

# Crisis resources sidebar
with st.sidebar:
    st.markdown("### 🆘 Crisis Resources")
    st.markdown("""
    **If you're in crisis, please reach out immediately:**

    🇺🇸 **National Suicide Prevention Lifeline**  
    📞 **988** (24/7, free, confidential)

    💬 **Crisis Text Line**  
    Text **HOME** to **741741**

    🚨 **Emergency Services: 911**

    🌍 **International Resources:**
    - UK: 116 123 (Samaritans)
    - Canada: 1-833-456-4566
    - Australia: 13 11 14
    """)

    st.markdown("---")
    st.markdown("### ℹ️ About This Chatbot")
    st.markdown("""
    This chatbot provides support based on real experiences shared in Reddit mental health communities. 

    **Important:** This is not a substitute for professional mental health care.
    """)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = """Hi! I'm here to provide support based on real experiences from mental health communities. 

I can help with topics like:
• Anxiety and stress management
• Depression support strategies  
• Therapy experiences
• Self-care and wellness tips
• Relationship and work stress

**Important:** I'm not a replacement for professional help. If you're in crisis, please use the resources in the sidebar immediately.

How can I support you today?"""

    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome_msg,
        "timestamp": datetime.now()
    })

# Display chat messages
st.markdown("### 💬 Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("is_crisis"):
            st.markdown("""
            <div class="crisis-warning">
            🚨 <strong>Crisis Support Detected</strong><br>
            Please consider reaching out to crisis resources immediately. Your safety is the top priority.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(message["content"])

        # Show metadata for assistant messages
        if message["role"] == "assistant" and "sources_used" in message:
            with st.expander("ℹ️ Response Details", expanded=False):
                st.write(f"📚 Based on {message['sources_used']} community discussions")
                st.write(f"🎯 Crisis Score: {message.get('crisis_score', 0):.2f}")
                st.write(f"⏰ Generated: {message['timestamp'].strftime('%Y-%m-%d %H:%M')}")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now()
    })

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Make API call to your RAG system
                response = requests.post(
                    "http://localhost:8000/query",
                    json={
                        "query": prompt,
                        "user_id": "streamlit_test_user"
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display crisis warning if detected
                    if result.get("is_crisis", False):
                        st.markdown("""
                        <div class="crisis-warning">
                        🚨 <strong>Crisis Support Resources</strong><br>
                        • <strong>National Suicide Prevention Lifeline: 988</strong><br>
                        • <strong>Crisis Text Line: Text HOME to 741741</strong><br>
                        • <strong>Emergency Services: 911</strong>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display the response
                    assistant_response = result.get("response",
                                                    "I apologize, but I couldn't generate a proper response.")
                    st.markdown(assistant_response)

                    # Add to chat history with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": datetime.now(),
                        "is_crisis": result.get("is_crisis", False),
                        "crisis_score": result.get("crisis_score", 0.0),
                        "sources_used": result.get("sources_used", 0)
                    })

                    # Show response metadata
                    with st.expander("ℹ️ Response Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sources Used", result.get("sources_used", 0))
                        with col2:
                            st.metric("Crisis Score", f"{result.get('crisis_score', 0):.2f}")
                        with col3:
                            status = "⚠️ Crisis" if result.get("is_crisis") else "✅ Normal"
                            st.metric("Status", status)

                else:
                    error_msg = f"API Error: {response.status_code}. Please make sure the RAG API server is running on http://localhost:8000"
                    st.error(error_msg)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })

            except requests.exceptions.ConnectionError:
                error_msg = """🔌 **Connection Error**: Cannot connect to the RAG API server.

**To fix this:**
1. Make sure you've completed Phase 2 (data processing)
2. Start the RAG API server by running: `python rag_query_engine.py`  
3. Wait for the server to start (you should see "Uvicorn running on http://0.0.0.0:8000")
4. Try your message again

**Alternative for testing:** You can test the basic interface without the API - just check the sidebar resources and UI layout."""

                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })

            except requests.exceptions.Timeout:
                error_msg = "⏰ Request timed out. The server might be processing. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })

            except Exception as e:
                error_msg = f"❌ Unexpected error: {str(e)}. If you're in crisis, please use the resources in the sidebar immediately."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
<p><strong>Disclaimer:</strong> This chatbot is for informational and support purposes only. It is not a substitute for professional mental health treatment, diagnosis, or advice. If you are experiencing a mental health crisis, please contact emergency services or a crisis hotline immediately.</p>

<p><strong>Privacy:</strong> Your conversations are used to improve the service but are not stored with personally identifiable information.</p>

<p><strong>Data Source:</strong> Responses are based on anonymized community discussions from Reddit mental health support communities.</p>
</div>
""", unsafe_allow_html=True)

# Quick test section in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧪 Quick Test")

    if st.button("Test API Connection"):
        try:
            test_response = requests.get("http://localhost:8000/health", timeout=5)
            if test_response.status_code == 200:
                st.success("✅ RAG API is running!")
                data = test_response.json()
                st.json(data)
            else:
                st.error(f"❌ API returned status: {test_response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure it's running on port 8000.")
        except Exception as e:
            st.error(f"❌ Test failed: {e}")

    st.markdown("### 📝 Sample Queries")
    sample_queries = [
        "I've been feeling anxious about work lately",
        "How do people cope with depression?",
        "What are some good self-care strategies?",
        "I'm having trouble sleeping due to stress",
        "How can I support a friend with mental health issues?"
    ]

    for query in sample_queries:
        if st.button(f"Try: '{query[:30]}...'", key=query):
            # Add the sample query to chat
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now()
            })
            st.rerun()

# Session statistics
if len(st.session_state.messages) > 1:  # More than just welcome message
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in st.session_state.messages if
                              msg["role"] == "assistant" and "sources_used" in msg]

        st.write(f"Messages: {len(user_messages)}")
        if assistant_messages:
            avg_sources = sum(msg.get("sources_used", 0) for msg in assistant_messages) / len(assistant_messages)
            st.write(f"Avg Sources: {avg_sources:.1f}")

            crisis_detected = sum(1 for msg in assistant_messages if msg.get("is_crisis", False))
            st.write(f"Crisis Alerts: {crisis_detected}")

# Clear chat button
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()