## 🧠 NestMind

“Every thought creates a path. NestMind remembers the paths you choose.” ✨

An intelligent, context-aware AI chatbot that doesn’t just answer questions <br>
       " It Thinks, Remembers, and Grows with your conversations."

---

## 🌟 What is NestMind?

NestMind is a smart conversational AI built using Streamlit + Google Gemini API.<br>
It organizes conversations into sections & sub-sections, remembers past interactions using embeddings, and delivers empathetic, structured, and relevant responses.

Think of it as a second brain chatbot 🧠💬.

---

## 🚀 Key Features  

✅ Context-aware AI responses <Br>
✅ Semantic memory using embeddings <br>
✅ Section-based & nested conversations <br>
✅ Focus Mode for deep discussions <br>
✅ Multiple chats with sidebar navigation<br>
✅ Clean & interactive Streamlit UI<br>
✅ Modular and extendable architecture<br>

---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit – Frontend UI
- Google Gemini API – LLM engine
- Text Embeddings – Semantic memory
- dotenv – Environment management
- Tenacity – Reliable API retries

---

## 📁 Project Structure
```bash
nestmind/
│
├── app.py              🧠 Main Streamlit application
├── ai_client.py        🤖 Gemini API wrapper & retry logic
├── retrieval.py        📚 Context retrieval (extendable)
├── requirements.txt    📦 Project dependencies
├── .env                🔐 API keys (not pushed to GitHub)
```

---

## ⚙️ Installation & Setup

1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/nestmind.git
cd nestmind
```
2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv

venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Set Environment Variables
Create a .env file in the root folder:
```bash
GEMINI_API_KEY=your_api_key_here
```

---

▶️ Run the Application

```bash
streamlit run app.py
```
🌐 Open browser → http://localhost:8501

---

## 💡 How NestMind Works

🧩 User Input <br>
🧠 Memory Search (Embeddings) <br>
🔗 Relevant Context Retrieved <br>
🤖 Gemini Generates Response <br>
💾 New Knowledge Stored for Future Use<br>

---

## 🎯 Configuration Options
```bash
Setting	                           Description

GEMINI_MODEL	         →            AI model used for responses
EMBEDDING_MODEL	  →              Model for semantic memory
SIMILARITY_THRESHOLD	  →               Memory match accuracy
PERSIST	         →              Enable chat persistence
```

---

## 🔮 Future Enhancements

🚀 Advanced knowledge base (Vector DB)<br>
🎨 Enhanced frontend UI & animations<br>
👤 Multi-user support & authentication<br>
🎤 Voice input & output<br>
🌐 Cloud deployment<br>

---

## ❤️ Why NestMind?

Because good chatbots answer,<br>
but great chatbots remember.

---

✨ Made with curiosity, logic, and a lot of thinking by Deveshree ✨