import json
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

"""
--------Functions--------
"""

def upload_files():
    print("\n📁 Please select your file(s)")
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Files")
    uploaded = {}
    for path in file_paths:
        with open(path, 'rb') as f:
            uploaded[os.path.basename(path)] = f.read()
    return uploaded, file_paths

def chunking():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter

def file_detection(splitter, uploaded, file_paths):
    all_pages = []
    all_chunks = []
    for file_name, file_path in zip(uploaded.keys(), file_paths):
        print(f"⏳ Loading: {file_name}")
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            print(f"\nUnsupported file type: {file_name}")
            continue
        pages = loader.load()
        all_pages.extend(pages)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
    print(f"\n✅ Total pages loaded: {len(all_pages)}")
    print(f"✅ Total chunks created: {len(all_chunks)}\n")
    return all_chunks, all_pages

def embedding_vectorization(all_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstores = FAISS.from_documents(documents=all_chunks, embedding=embedding_model)
    vectorstores.save_local("myvectorstore")
    for i, chunk in enumerate(all_chunks[:5]):
        print(f"--- Chunk {i+1} ---\n{chunk.page_content[:200]}")
    vectorstore = FAISS.load_local("myvectorstore", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore, embedding_model

def retriaval(vectorstore):
    query_list = []
    results_list = []
    while True:
        query = input("Ask your question (type 'exit' to finish): ").lower()
        if query == "exit":
            break
        query_list.append(query)
    number_of_chunks = int(input("How many chunks do you want for each answer? : "))
    for question in query_list:
        results = vectorstore.similarity_search(question, k=number_of_chunks)
        results_list.append(results)
    return results_list, query_list, number_of_chunks

class RAGEnv(gym.Env):
    def __init__(self, vectorstore, embedding_model, query_list, top_k):
        super(RAGEnv, self).__init__()
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.query_list = query_list
        self.top_k = top_k
        self.current_idx = 0
        self.action_space = spaces.Discrete(self.top_k)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(384,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_idx = random.randint(0, len(self.query_list) - 1)
        self.current_question = self.query_list[self.current_idx]
        self.query_vector = self.embedding_model.embed_query(self.current_question)
        self.candidates = self.vectorstore.similarity_search(self.current_question, k=self.top_k)
        return np.array(self.query_vector, dtype=np.float32), {}

    def step(self, action):
        selected_doc = self.candidates[action]
        print(f"\n🧠 Question:  {self.current_question}")
        print(f"📚 Answer:  {selected_doc.page_content[:300]}")
        feedback = input("🤖 Was this answer good? (yes / no / maybe): ").lower()
        if feedback == "yes": reward = 1
        elif feedback == "no": reward = -1
        elif feedback == "maybe": reward = 0
        else:
            print("⚠ Wrong feedback. Reward considered as 0.")
            reward = 0
        return np.array(self.query_vector, dtype=np.float32), reward, True, False, {}

class QTableAgent:
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, decay=0.99):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.decay = decay
        self.action_size = action_size

    def get_state_key(self, question):
        return question.strip().lower()

    def choose_action(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                "original_question": state_key,
                "chunk_scores": [0.0] * self.action_size,
                "best_answer": "",
                "source_file": ""
            }
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state_key]["chunk_scores"]))

    def update(self, state_key, action, reward, answer=None, source_file=None, original_question=None):
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                "original_question": original_question,
                "chunk_scores": [0.0] * self.action_size,
                "best_answer": "",
                "source_file": ""
            }
        chunk_scores = self.q_table[state_key]["chunk_scores"]
        action = int(action)
        q_old = chunk_scores[action]
        q_new = q_old + self.lr * (reward - q_old)
        self.q_table[state_key]["chunk_scores"][action] = q_new
        if reward == 1 and answer and source_file:
            self.q_table[state_key]["best_answer"] = answer
            self.q_table[state_key]["source_file"] = source_file
        self.epsilon *= self.decay

    def save(self, name_location="q_table.json"):
        with open(name_location, "w") as f:
            json.dump(self.q_table, f)

    def load(self, name_location="q_table.json"):
        with open(name_location, "r") as f:
            self.q_table = json.load(f)

"""
-------- Main Program --------
"""
uploaded, file_paths = upload_files()
splitter = chunking()
all_chunks, all_pages = file_detection(splitter, uploaded, file_paths)
vectorstore, embedding_model = embedding_vectorization(all_chunks)
results_list, query_list, number_of_chunks = retriaval(vectorstore)

env = RAGEnv(vectorstore, embedding_model, query_list, top_k=number_of_chunks)
action_size = number_of_chunks
agent = QTableAgent(action_size)

if os.path.exists("q_table.json"):
    agent.load()
    print("✅ Q-table loaded from file.")
else:
    print("⚠ No existing Q-table. Starting fresh.")

print("\n🚀 Starting agent training...\n")
for i in range(len(query_list)):
    obs, _ = env.reset()
    state_key = env.current_question.strip().lower()
    original_question = env.current_question
    action = agent.choose_action(state_key)
    obs, reward, terminated, truncated, info = env.step(action)
    selected_answer = env.candidates[action].page_content[:300]
    source_file = list(uploaded.keys())[0] if uploaded else "Unknown"
    agent.update(state_key, action, reward, selected_answer, source_file, original_question)
    print(f"\n🎯 Action: {action} | ✅ Reward: {reward} | 📉 Epsilon: {agent.epsilon:.4f}")
    print("-" * 50)

agent.save()
print("\n✅ Agent Q-table saved to q_table.json\n")
