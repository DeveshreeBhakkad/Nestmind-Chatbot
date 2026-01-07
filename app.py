# app.py

# app.py

import os
import time
import uuid
import math
import streamlit as st
from dotenv import load_dotenv
from ai_client import call_openai_chat

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="NestMind",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# GLOBAL CSS (Sidebar smaller, spacing tighter)
# ======================================================
st.markdown("""
<style>
/* Reduce sidebar width */
[data-testid="stSidebar"] {
    width: 260px;
}

/* App background */
.stApp {
    background-color: #0b0b0b;
}

/* Title */
h1 {
    color: #facc15;
    margin-top: -40px;
}

/* Subtitle */
.subtitle {
    color: #d6d3d1;
    text-align: center;
    font-size: 16px;
    margin-top: -10px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1c1917;
    border-right: 1px solid #292524;
}

/* Chat bubbles */
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

/* Buttons */
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
# CHAT DATA STRUCTURE
# ======================================================
def create_chat(title="New Chat"):
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "messages": [
            {"role": "assistant", "content": "Ask me anything."}
        ]
    }

# ======================================================
# SESSION STATE INIT
# ======================================================
if "chats" not in st.session_state:
    first = create_chat("Welcome")
    st.session_state.chats = {first["id"]: first}
    st.session_state.active_chat_id = first["id"]

chats = st.session_state.chats
active_id = st.session_state.active_chat_id
active_chat = chats[active_id]

# ======================================================
# SIDEBAR — CHAT MANAGER (VS CODE STYLE)
# ======================================================
with st.sidebar:
    st.markdown("### 🧠 NestMind Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new = create_chat()
        chats[new["id"]] = new
        st.session_state.active_chat_id = new["id"]
        st.rerun()

    st.markdown("---")

    for cid, chat in chats.items():
        col1, col2 = st.columns([6, 1])

        with col1:
            if st.button(
                f"📄 {chat['title']}",
                key=f"open_{cid}",
                use_container_width=True
            ):
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
    if new_title != active_chat["title"]:
        active_chat["title"] = new_title

# ======================================================
# MAIN HEADER (SHIFTED UP)
# ======================================================
st.markdown("""
<h1 style="text-align:center;">NestMind</h1>
<p class="subtitle">
Memory-aware AI for focused conversations
</p>
""", unsafe_allow_html=True)

# ======================================================
# CHAT DISPLAY
# ======================================================
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ======================================================
# INPUT
# ======================================================
user_input = st.chat_input("Ask NestMind…")

if user_input:
    active_chat["messages"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    messages = [{"role": "system", "content": "You are NestMind, thoughtful and concise."}]
    messages += active_chat["messages"]

    reply = call_openai_chat(
        messages,
        max_tokens=400,
        temperature=0.3
    )

    active_chat["messages"].append(
        {"role": "assistant", "content": reply}
    )

    with st.chat_message("assistant"):
        st.write(reply)
