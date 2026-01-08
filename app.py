# app.py

import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from ai_client import call_openai_chat

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Nested Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# GLOBAL CSS
# ======================================================
st.markdown("""
<style>
.stApp {
    background-color: #0b0b0b;
}

h1 {
    color: #facc15;
    margin-top: -40px;
}

.subtitle {
    color: #d6d3d1;
    text-align: center;
    font-size: 16px;
    margin-top: -8px;
}

[data-testid="stSidebar"] {
    background-color: #1c1917;
    border-right: 1px solid #292524;
    width: 260px;
}

[data-testid="stChatMessage"] {
    background-color: #1c1917;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #292524;
}

[data-testid="stChatMessage"][data-role="user"] {
    border-left: 4px solid #facc15;
}

[data-testid="stChatMessage"][data-role="assistant"] {
    border-left: 4px solid #a16207;
}

button {
    background-color: #a16207 !important;
    color: black !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

button:hover {
    background-color: #facc15 !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# ENV CHECK
# ======================================================
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ GEMINI_API_KEY missing")
    st.stop()

# ======================================================
# CHAT STRUCTURE
# ======================================================
def create_chat(title="New Chat"):
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "messages": []
    }

# ======================================================
# SESSION STATE INIT
# ======================================================
if "chats" not in st.session_state:
    first = create_chat("Welcome")
    st.session_state.chats = {first["id"]: first}
    st.session_state.active_chat_id = first["id"]

if "resource_tab" not in st.session_state:
    st.session_state.resource_tab = None

chats = st.session_state.chats
active_chat = chats[st.session_state.active_chat_id]

# ======================================================
# SIDEBAR — CHAT MANAGER (WITH RENAME & DELETE)
# ======================================================
with st.sidebar:
    st.markdown("### 🧠 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new = create_chat()
        chats[new["id"]] = new
        st.session_state.active_chat_id = new["id"]
        st.rerun()

    st.markdown("---")

    for cid, chat in chats.items():
        col1, col2 = st.columns([6, 1])

        with col1:
            if st.button(f"📄 {chat['title']}", key=f"open_{cid}", use_container_width=True):
                st.session_state.active_chat_id = cid
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                if len(chats) > 1:
                    del chats[cid]
                    st.session_state.active_chat_id = list(chats.keys())[0]
                    st.rerun()

    st.markdown("---")

    new_title = st.text_input("✏️ Rename chat", value=active_chat["title"])
    if new_title.strip() and new_title != active_chat["title"]:
        active_chat["title"] = new_title

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<h1 style="text-align:center;">NestMind</h1>
<p class="subtitle">Nested, contextual conversations powered by AI</p>
""", unsafe_allow_html=True)

# ======================================================
# RESOURCE SECTION
# ======================================================
st.markdown("### 📌 Explore Resources")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔗 Links", use_container_width=True):
        st.session_state.resource_tab = "links"

with col2:
    if st.button("🖼️ Images", use_container_width=True):
        st.session_state.resource_tab = "images"

with col3:
    if st.button("📄 PDFs", use_container_width=True):
        st.session_state.resource_tab = "pdfs"

if st.session_state.resource_tab == "links":
    st.markdown("""
    #### 🔗 Useful Links
    - https://streamlit.io
    - https://ai.google.dev
    - https://github.com
    """)

elif st.session_state.resource_tab == "images":
    st.markdown("#### 🖼️ Reference Image")
    st.image(
        "https://images.unsplash.com/photo-1677442136019-21780ecad995",
        use_column_width=True
    )

elif st.session_state.resource_tab == "pdfs":
    st.markdown("""
    #### 📄 PDFs
    - AI System Design Notes (coming soon)
    - Context-Aware Chat Architectures
    """)

st.markdown("---")

# ======================================================
# CHAT UI
# ======================================================
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask NestMind…")

if user_input:
    active_chat["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    messages = [
        {"role": "system", "content": "You are NestMind, a thoughtful and concise AI assistant."}
    ] + active_chat["messages"]

    reply = call_openai_chat(
        messages,
        max_tokens=400,
        temperature=0.3
    )

    active_chat["messages"].append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)
