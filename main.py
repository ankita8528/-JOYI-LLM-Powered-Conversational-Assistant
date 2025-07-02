import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from memory_management import get_memory
from prompt import prompt_template, get_trimmer
from model import get_model
from workflow import get_workflow

# Load environment variables (replace with st.secrets or .env as needed)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY")
os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY")

# Initialize chatbot components
model = get_model()
memory = get_memory()
trimmer = get_trimmer(model)
workflow = get_workflow(model, prompt_template, trimmer)
app = workflow.compile(checkpointer=memory)

# Streamlit app config
st.set_page_config(page_title="JOYI - Your AI Assistant", layout="wide")
st.title("🤖 JOYI - Your Helpful AI Companion")
st.caption("Powered by LangGraph + Google Gemini")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Optional welcome message
if not st.session_state.chat_history:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("**JOYI:** Hello! I'm JOYI. Ask me anything 🌟")

# Display full chat history
for msg in st.session_state.chat_history:
    role = "YOU" if isinstance(msg, HumanMessage) else "JOYI"
    avatar = "🧑‍💻" if role == "YOU" else "🤖"
    with st.chat_message(role.lower(), avatar=avatar):
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #1e1e1e; color: #fff;">
            <strong>{role}:</strong> {msg.content.strip().replace('\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)

# User input
user_input = st.chat_input("Talk to JOYI...")
if user_input:
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #2d2d2d; color: #fff;">
            <strong>YOU:</strong> {user_input.strip()}
        </div>
        """, unsafe_allow_html=True)

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    input_state = {
        "messages": st.session_state.chat_history,
        "language": "english"
    }

    # Stream and display AI response
    result = app.invoke(input_state, config={"configurable": {"thread_id": "joyi-session"}})
    response_stream = list(result["messages"])

    # Take only the latest AI message (last one in the stream)
    latest_response = response_stream[-1].content.strip()

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"""<div style="padding: 1rem; border-radius: 0.5rem; background-color: #1e1e1e; color: #fff;"><strong>JOYI:</strong> {latest_response.replace('\n', '<br>')}</div>""", unsafe_allow_html=True)
    st.session_state.chat_history.append(AIMessage(content=latest_response))

# Reset button
if st.button("🔄 Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()
