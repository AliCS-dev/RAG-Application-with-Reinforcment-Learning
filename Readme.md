# 📄 RAG Application with Reinforcement Learning Agent

---

## 🧠 Project Title

**RAG-Based Document QA with Reinforcement Learning (RL)**  
A Retrieval-Augmented Generation (RAG) system powered by FAISS, LangChain, and a custom Q-learning agent for feedback-driven learning. Designed for querying uploaded documents interactively and improving answer selection using reinforcement learning.

---

## 👥 Team

This project was developed at **Aalto University Summer School 2025** by:

- `ali_cs_dev`
- `muham`
- Tunc Saracoglu
- Arjun
- Bharat 

---

## 📦 Dependencies

Install all required Python packages:

```bash
pip install langchain langchain-community faiss-cpu gymnasium sentence-transformers
pip install unstructured python-docx
These libraries support:

Document loading (PDF, DOCX, TXT)

Text chunking & vector embedding

FAISS vector storage & retrieval

Reinforcement learning with Gymnasium

User feedback collection via terminal

Sentence transformers for semantic similarity

📁 File Structure
bash
Copy
Edit
RAG_Application/
│
├── App.py               # Main Python script
├── q_table.json         # Q-learning saved state (after training)
├── myvectorstore/       # FAISS vectorstore (auto-generated)
├── README.md            # You're here
🏗️ Workflow Overview
File Upload (via file dialog)

Document Loading & Chunking

Embedding & FAISS Index Creation

User Queries & Initial Retrieval

Interactive Reinforcement Learning Agent

User Feedback Collection (reward system)

Q-table Update & Saving

🔍 Detailed Component Breakdown
1. File Upload (Local)
Uses tkinter to allow local file selection:

.pdf, .docx, and .txt supported

python
Copy
Edit
def upload_files()
2. Text Chunking
Chunks are created using LangChain’s RecursiveCharacterTextSplitter.

chunk_size=500

chunk_overlap=100

python
Copy
Edit
def chunking()
3. Document Detection & Chunking
Each file is loaded via LangChain’s loaders:

PyPDFLoader

TextLoader

UnstructuredWordDocumentLoader

python
Copy
Edit
def file_detection(splitter, uploaded, file_paths)
4. Embedding & FAISS Vectorstore
Creates vector embeddings using HuggingFace model:

all-MiniLM-L6-v2

Stores vectors in FAISS

Saves index to local disk

python
Copy
Edit
def embedding_vectorization(all_chunks)
5. User Query Collection
Prompts user to enter questions until "exit" is typed.

Performs similarity search for each query

User defines k (number of top chunks to retrieve)

python
Copy
Edit
def retriaval(vectorstore)
6. RAG Environment (RAGEnv)
A custom Gymnasium environment:

Observation: 384-dim embedding vector

Action: Choose one of k chunks

Reward: Based on user input (yes=+1, no=-1, maybe=0)

Terminates: After one step per query

python
Copy
Edit
class RAGEnv(gym.Env)
7. Q-Learning Agent (QTableAgent)
Agent maintains a Q-table per question:

Tracks chunk scores

Chooses actions using epsilon-greedy

Updates scores based on user feedback

Saves/loads Q-table to/from JSON

python
Copy
Edit
class QTableAgent
8. Training Loop
For each question:

Resets environment

Selects action via choose_action()

Calls step()

Updates Q-table via update()

Displays metrics

🎮 Reward System
Feedback Input	Reward
yes	+1
no	-1
maybe	0

Used to adjust Q-values for specific (question, chunk) pairs.

💾 Persistence
FAISS index is saved as myvectorstore/

Q-table is saved as q_table.json

These files allow the agent to resume learning or re-use stored knowledge.

🧪 Example Usage (Terminal)
pgsql
Copy
Edit
$ python App.py

📁 Please select your file(s)
⏳ Loading: your_file.pdf
✅ Total pages loaded: 10
✅ Total chunks created: 25

Ask your question (type 'exit' to finish): what is quantum computing?
Ask your question (type 'exit' to finish): exit

How many chunks do you want for each answer? : 3

🧠 Question: what is quantum computing?
📚 Answer: (top FAISS result content)

🤖 Was this answer good? (yes / no / maybe): yes

🎯 Action: 2 | ✅ Reward: 1 | 📉 Epsilon: 0.99
--------------------------------------------------
💡 Tips
Run in a virtual environment to avoid dependency issues.

Train the agent with multiple questions to improve learning.

Save your Q-table regularly.

🛠️ Future Enhancements
Replace terminal input with a GUI (e.g. Streamlit or Gradio)

Add support for multiple file tracking in Q-table

Introduce confidence scoring using LLMs

Expand action space with answer re-ranking

🧠 Academic Note
This project is an academic experiment blending:

Information Retrieval (IR)

Reinforcement Learning (RL)

Natural Language Processing (NLP)

Ideal for learning how retrieval-based QA systems can be improved with user feedback.

```

