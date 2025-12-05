import os
import json
import math
import time
import glob
import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html as st_html

# PAGE CONFIG MUST COME BEFORE ANY STREAMLIT UI
st.set_page_config(page_title="nestmind", layout="wide")

from ai_client import call_openai_chat

# ==================
# GLOBAL CONFIG
# ==================
GEMINI_MODEL = "models/gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
SIMILARITY_THRESHOLD = 0.75
MAX_HISTORY_MESSAGES = None   # None => send full history to LLM (unlimited)

# 🔒 Persistence toggle:
# False => fresh app every run (no background history)
# True  => save/load chat_<slug>.json files
PERSIST = False

# ==================
# ENV / KEY
# ==================
def load_dotenv_and_key():
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or ""
    if len(key) >= 2 and ((key[0] == '"' and key[-1] == '"') or (key[0] == "'" and key[-1] == "'")):
        key = key[1:-1]
    return key

api_key = load_dotenv_and_key()
if not api_key:
    st.error("GEMINI_API_KEY not found. Put GEMINI_API_KEY=... in a .env file next to app.py.")
    st.stop()

st.sidebar.success("✅ GEMINI_API_KEY loaded (sanitized).")

# ==================
# Embeddings (Gemini)
# ==================
import google.generativeai as genai
genai.configure(api_key=api_key)

def get_embedding(text: str):
    try:
        out = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        emb = out.get("embedding")
        return emb["values"] if isinstance(emb, dict) and "values" in emb else emb
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# ==================
# Similarity / Memory
# ==================
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def find_best_memory(query: str, data: dict):
    q_emb = get_embedding(query)
    if q_emb is None:
        return None, 0.0
    best_id, best_score = None, -1.0
    for nid, node in data.items():
        emb = node.get("embedding")
        if emb:
            s = cosine_similarity(q_emb, emb)
            if s > best_score:
                best_id, best_score = nid, s
    return best_id, best_score

# ==================
# LLM wrapper
# ==================
SYSTEM_PROMPT = (
    "You are NestMind, an intelligent and empathetic AI assistant. "
    "Be helpful, concise but thorough, and explain steps where useful. "
    "If the user asks for code, give runnable code and explain. "
    "If provided relevant memory below, incorporate it into your answer."
)

def get_llm_response(history_messages, memory_text=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory_text:
        msgs.append({"role": "system", "content": f"Relevant memory: {memory_text}"})

    iterable = history_messages if MAX_HISTORY_MESSAGES is None else history_messages[-MAX_HISTORY_MESSAGES:]
    for m in iterable:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})

    try:
        return call_openai_chat(msgs, model=GEMINI_MODEL, max_tokens=900, temperature=0.2)
    except Exception as e:
        return f"Gemini API Error: {e}"

# ===============================
# Chats + Stacked Sections (UI)
# ===============================
def slugify(name: str) -> str:
    s = "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch in ("-", "_"))
    return s or f"chat_{int(time.time())}"

def chat_file(slug: str) -> str:
    return f"chat_{slug}.json"

def _new_section(title: str, topic_label: str = "") -> dict:
    cd = {
        "start": {"message": f"{('Topic **'+topic_label+'** — ') if topic_label else ''}Ask me anything!",
                  "responses": {}, "embedding": None}
    }
    return {
        "id": f"sec_{int(time.time()*1000)}",
        "title": title.strip() or "untitled section",
        "messages": [{"role": "assistant", "content": cd["start"]["message"]}],
        "conversation_data": cd,
        "current_node": "start",
        "children": []
    }

def _sanitize_chat(chat: dict) -> dict:
    sections = chat.get("sections", {})
    chat["roots"] = [rid for rid in chat.get("roots", []) if rid in sections]
    for n in sections.values():
        n["children"] = [cid for cid in n.get("children", []) if cid in sections]
    if chat.get("selected_section") not in sections:
        chat["selected_section"] = None
        chat["focus_mode"] = False
    return chat

def load_chat(slug: str) -> dict:
    if PERSIST:
        path = chat_file(slug)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chat = json.load(f)
                return _sanitize_chat(chat)
            except json.JSONDecodeError:
                st.error(f"{path} is malformed. Back it up and delete to reinitialize.")
    root = _new_section(f"{slug} — section 1", topic_label=slug)
    return {
        "meta": {"title": slug, "slug": slug},
        "sections": {root["id"]: root},
        "roots": [root["id"]],
        "selected_section": None,
        "focus_mode": False
    }

def save_chat(chat: dict):
    if not PERSIST:
        return
    slug = chat["meta"]["slug"]
    try:
        with open(chat_file(slug), "w", encoding="utf-8") as f:
            json.dump(chat, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save {chat_file(slug)}: {e}")

def collect_subtree_ids(chat: dict, section_id: str):
    if section_id not in chat["sections"]:
        return []
    ids = [section_id]
    for cid in chat["sections"][section_id].get("children", []):
        ids.extend(collect_subtree_ids(chat, cid))
    return ids

def remove_section(chat: dict, section_id: str):
    if section_id not in chat["sections"]:
        return
    to_delete = set(collect_subtree_ids(chat, section_id))

    for sid, node in list(chat["sections"].items()):
        if "children" in node:
            node["children"] = [cid for cid in node["children"] if cid not in to_delete]

    chat["roots"] = [rid for rid in chat["roots"] if rid not in to_delete]

    for tid in to_delete:
        chat["sections"].pop(tid, None)

    if not chat["roots"]:
        newroot = _new_section(f"{chat['meta']['title']} — section 1", topic_label=chat['meta']['title'])
        chat["sections"][newroot["id"]] = newroot
        chat["roots"].append(newroot["id"])

    if chat.get("selected_section") in to_delete:
        chat["selected_section"] = None
        chat["focus_mode"] = False

def chat_one_turn(section_node: dict, user_text: str, chat_title: str) -> str:
    cd = section_node["conversation_data"]
    current_node = section_node["current_node"]

    next_node = cd.get(current_node, {}).get("responses", {}).get(user_text.lower())
    if next_node:
        st.toast("Memory hit (exact) ✅", icon="📚")
        section_node["current_node"] = next_node
        return cd[next_node]["message"]

    best_node, best_score = find_best_memory(user_text, cd)
    if best_node and best_score >= SIMILARITY_THRESHOLD:
        st.toast(f"Memory hit (semantic) — score {best_score:.2f}", icon="🧠")
        section_node["current_node"] = best_node
        return cd[best_node]["message"]

    memory_text = cd[best_node]["message"] if (best_node and best_score > 0.45) else None
    chat_history = section_node["messages"] + [{"role": "user", "content": user_text}]
    bot = get_llm_response(chat_history, memory_text=memory_text)

    if isinstance(bot, str) and not bot.startswith("Gemini API Error"):
        newid = f"node_{len(cd)}" if f"node_{len(cd)}" not in cd else f"node_{len(cd)+1}"
        emb = get_embedding(bot)
        cd[newid] = {"message": bot, "responses": {}, "embedding": emb}
        cd.setdefault(current_node, {}).setdefault("responses", {})[user_text.lower()] = newid
        st.toast(f"✅ Saved to **{chat_title}** memory", icon="💾")
    return bot

# =========== 
# PAGE SHELL
# ===========

st.title("nestmind")

if "chat_list" not in st.session_state:
    st.session_state.chat_list = []
if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "last_selected_chat" not in st.session_state:
    st.session_state.last_selected_chat = None
if "scroll_to" not in st.session_state:
    st.session_state.scroll_to = None

# -----------------
# Sidebar controls
# -----------------
with st.sidebar:
    st.subheader("Chats")

    for nm in st.session_state.chat_list:
        sl = slugify(nm)
        if sl not in st.session_state.chats:
            st.session_state.chats[sl] = load_chat(sl)

    if st.session_state.chat_list:
        opts = [slugify(x) for x in st.session_state.chat_list]
        if st.session_state.selected_chat not in opts:
            st.session_state.selected_chat = opts[0]
        chosen = st.radio(
            "Select chat",
            options=opts,
            format_func=lambda s: st.session_state.chats[s]["meta"]["title"],
            index=opts.index(st.session_state.selected_chat) if st.session_state.selected_chat in opts else 0
        )
        if chosen != st.session_state.selected_chat:
            st.session_state.selected_chat = chosen

        if st.button("🗑️ Delete chat"):
            slug = st.session_state.selected_chat
            try:
                title = st.session_state.chats[slug]["meta"]["title"]
            except Exception:
                title = slug
            st.session_state.chat_list = [c for c in st.session_state.chat_list if slugify(c) != slug]
            st.session_state.chats.pop(slug, None)
            if PERSIST:
                try:
                    os.remove(chat_file(slug))
                except FileNotFoundError:
                    pass
            st.toast(f"Deleted chat **{title}**", icon="🗑️")
            if st.session_state.chat_list:
                st.session_state.selected_chat = slugify(st.session_state.chat_list[0])
            else:
                st.session_state.selected_chat = None
            st.rerun()

    st.divider()
    new_chat_name = st.text_input("New chat name", placeholder="e.g., html, project-x, notes")
    if st.button("➕ New chat"):
        nm = new_chat_name.strip() or f"chat_{len(st.session_state.chat_list)+1}"
        sl = slugify(nm)
        if sl in st.session_state.chats:
            st.warning("Chat already exists.")
        else:
            st.session_state.chat_list.append(nm)
            st.session_state.chats[sl] = load_chat(sl)
            st.session_state.selected_chat = sl
            st.success(f"Created chat '{nm}'")

if st.session_state.selected_chat and st.session_state.selected_chat != st.session_state.last_selected_chat:
    title = st.session_state.chats[st.session_state.selected_chat]["meta"]["title"]
    st.toast(f"🧠 You are in **{title}**", icon="🎯")
    st.session_state.last_selected_chat = st.session_state.selected_chat

if not st.session_state.chat_list or not st.session_state.selected_chat:
    st.info("No chats yet. Use the sidebar to create one.")
    st.stop()

chat = st.session_state.chats[st.session_state.selected_chat]

title_col, spacer, add_col = st.columns([6, 1, 3], vertical_alignment="center")
with title_col:
    st.subheader(f"Chat: {chat['meta']['title']}")
with add_col:
    with st.expander("➕ Add section", expanded=False):
        sec_title = st.text_input("Section title", key=f"addsec_{chat['meta']['slug']}")
        if st.button("Add section"):
            sec = _new_section(sec_title or f"{chat['meta']['title']} — section {len(chat['roots'])+1}",
                               topic_label=chat['meta']['title'])
            chat["sections"][sec["id"]] = sec
            chat["roots"].append(sec["id"])
            save_chat(chat)
            st.rerun()

st.markdown("---")

def render_section(chat: dict, section_id: str, depth: int = 0):
    if section_id not in chat["sections"]:
        return
    node = chat["sections"][section_id]

    st.markdown(f"<div id='anchor-{section_id}'></div>", unsafe_allow_html=True)

    pad_css = f"margin-left:{min(depth*18, 120)}px"
    with st.container(border=True):
        st.markdown(f"<div style='{pad_css}'></div>", unsafe_allow_html=True)
        hdr_col, act_col = st.columns([6, 4])
        with hdr_col:
            focused_here = chat.get("focus_mode") and chat.get("selected_section") == section_id
            badge = " <span style='padding:2px 8px;border-radius:10px;background:#3b82f6;color:white;font-size:0.8rem;'>FOCUSED</span>" if focused_here else ""
            st.markdown(f"### {node['title']}{badge}", unsafe_allow_html=True)

        with act_col:
            c1, c2, c3 = st.columns(3)
            focus_label = "🔦 Focus (ON)" if (chat.get("focus_mode") and chat.get("selected_section") == section_id) else "Focus"
            if c1.button(focus_label, key=f"focus_{section_id}"):
                if chat.get("focus_mode") and chat.get("selected_section") == section_id:
                    chat["focus_mode"] = False
                    st.session_state.scroll_to = section_id
                else:
                    chat["selected_section"] = section_id
                    chat["focus_mode"] = True
                save_chat(chat)
                st.rerun()

            if c2.button("➕ Add sub-section", key=f"addchild_{section_id}"):
                child = _new_section(f"{node['title']} — sub", topic_label=chat['meta']['title'])
                node["children"].append(child["id"])
                chat["sections"][child["id"]] = child
                save_chat(chat)
                st.rerun()

            if c3.button("🗑️ Delete", key=f"del_{section_id}"):
                remove_section(chat, section_id)
                save_chat(chat)
                st.rerun()

        for m in node["messages"]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        with st.form(key=f"form_{section_id}", clear_on_submit=True):
            u = st.text_input("Message", key=f"in_{section_id}", placeholder="Type here…")
            sent = st.form_submit_button("Send")

        if sent and u.strip():
            txt = u.strip()
            node["messages"].append({"role": "user", "content": txt})
            with st.chat_message("user"):
                st.write(txt)
            bot = chat_one_turn(node, txt, chat_title=chat['meta']['title'])
            save_chat(chat)
            with st.chat_message("assistant"):
                st.write(bot)
            node["messages"].append({"role": "assistant", "content": bot})

        for cid in list(node.get("children", [])):
            if cid in chat["sections"]:
                render_section(chat, cid, depth+1)

if chat.get("selected_section") not in chat["sections"]:
    chat["selected_section"] = None
    chat["focus_mode"] = False

if chat.get("focus_mode") and chat.get("selected_section"):
    render_section(chat, chat["selected_section"], depth=0)
else:
    chat["roots"] = [rid for rid in chat.get("roots", []) if rid in chat["sections"]]
    for rid in chat["roots"]:
        render_section(chat, rid, depth=0)

if st.session_state.get("scroll_to"):
    target = st.session_state.pop("scroll_to")
    st_html(f"""
        <script>
        const el = document.getElementById("anchor-{target}");
        if (el) {{
          el.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        </script>
    """, height=0)

st.markdown("---")
save_chat(chat)

