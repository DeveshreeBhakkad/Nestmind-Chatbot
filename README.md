## 🧠 NestMind

“Every thought creates a path. NestMind remembers the paths you choose.” ✨

An intelligent, context-aware AI chatbot that doesn’t just answer questions 
       " It Thinks, Remembers, and Grows with your conversations."

---

## 🌟 What is NestMind?

NestMind is a smart conversational AI built using Streamlit + Google Gemini API.
It organizes conversations into sections & sub-sections, remembers past interactions using embeddings, and delivers empathetic, structured, and relevant responses.

Think of it as a second brain chatbot 🧠💬.

---

## 🚀 Key Features  

- ✅ Context-aware AI responses 
- ✅ Semantic memory using embeddings
- ✅ Section-based & nested conversations
- ✅ Focus Mode for deep discussions 🔦
- ✅ Multiple chats with sidebar navigation
- ✅ Clean & interactive Streamlit UI
- ✅ Modular and extendable architecture

---

## 🛠️ Tech Stack

- 🔹 Python 3.x
- 🔹 Streamlit – Frontend UI
- 🔹 Google Gemini API – LLM engine
- 🔹 Text Embeddings – Semantic memory
- 🔹 dotenv – Environment management
- 🔹 Tenacity – Reliable API retries

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

- 🧩 User Input →
- 🧠 Memory Search (Embeddings) 
- 🔗 Relevant Context Retrieved 
- 🤖 Gemini Generates Response 
- 💾 New Knowledge Stored for Future Use

---

## 🎯 Configuration Options
```bash
Setting	                        Description

GEMINI_MODEL	         →           AI model used for responses
EMBEDDING_MODEL	  →           Model for semantic memory
SIMILARITY_THRESHOLD	  →           Memory match accuracy
PERSIST	         →           Enable chat persistence
```

---

## 🔮 Future Enhancements

-  🚀 Advanced knowledge base (Vector DB)
- 🎨 Enhanced frontend UI & animations
- 👤 Multi-user support & authentication
- 🎤 Voice input & output
- 🌐 Cloud deployment

---

## ❤️ Why NestMind?

Because good chatbots answer,/n
but great chatbots remember.

---

✨ Made with curiosity, logic, and a lot of thinking by Deveshree ✨