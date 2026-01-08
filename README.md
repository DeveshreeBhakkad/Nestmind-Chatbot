# 🧠 Nested Chatbot (NestMind)

> *“Not just a chatbot — a system that understands conversations in context.”*

🔗 **Live Demo:** https://nested-chatbot.streamlit.app

---

## 🌟 Overview

**Nested Chatbot (NestMind)** is a multi-chat, memory-aware AI assistant built to explore how conversations can be organized, contextual, and structured — similar to how humans think in nested ideas rather than flat chats.

The project focuses on **clean UX, real-world AI constraints, and production-ready behavior**, rather than just generating responses.

---

## ✨ Key Features

### 🗂️ Multi-Chat System (ChatGPT-style)
- Create multiple chats
- Switch between chats instantly
- Rename chats
- Delete chats
- Sidebar behaves like a **VS Code file explorer**

### 🧠 Context-Aware Conversations
- Each chat maintains its own conversation context
- System prompt ensures thoughtful, concise responses
- Designed for focused and meaningful interactions

### 🎨 Polished UI / UX
- Dark theme with warm accent colors
- Clean chat bubbles
- Minimal, distraction-free interface
- Optimized spacing for better readability

### 🛡️ Graceful API Limit Handling
- Handles Gemini API quota limits without crashing
- Shows a user-friendly message instead of errors
- Designed with **real-world production constraints** in mind

### ☁️ Deployed & Live
- Hosted on **Streamlit Cloud**
- Secure API key management using Secrets
- Public, shareable URL

---

## 🧰 Tech Stack

- **Frontend & App Framework:** Streamlit
- **Language:** Python 3.11
- **AI Model:** Google Gemini (Generative AI)
- **Deployment:** Streamlit Cloud
- **Version Control:** Git & GitHub

---

## 🚀 Getting Started (Local Setup)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/nested-chatbot.git
cd nested-chatbot
```

2️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Set API key

Create a .env file:
```bash
GEMINI_API_KEY=your_api_key_here
```
5️⃣ Run the app
```bash
streamlit run app.py
```
---

☁️ Deployment Notes

- The app is deployed using Streamlit Cloud

- API keys are stored securely using Secrets

- The app automatically redeploys on every GitHub push

- Designed to work reliably even when API limits are reached

---


🧠 Design Philosophy

This project intentionally prioritizes:

  - Stability over unlimited usage

  - Clear UX over flashy UI

  - Graceful failure over crashes

  - Real-world constraints over toy examples

---

🔮 Future Enhancements

- Persistent chat history (database-backed)

- Chat search functionality

- Memory toggles per chat

- User authentication

- Advanced memory visualization

---

👩‍💻 Author

Deveshree Bhakkad
Final-year AIML student | AI Systems & Product Thinking
🔗 GitHub: https://github.com/DeveshreeBhakkad

⭐ If you like this project, feel free to star the repo!


---

