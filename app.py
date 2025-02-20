import os
import fitz  # PyMuPDF
import re
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.retrievers.multi_query import MultiQueryRetriever
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# ========= 🔹 Load Policy Documents & Preprocess 🔹 =========
POLICY_DOCUMENT_PATHS = ["sompodom_merged.pdf", "Zurich_APAC.pdf"]
policy_texts = []

try:
    for file_path in POLICY_DOCUMENT_PATHS:
        with fitz.open(file_path) as doc:
            text = "\n".join([page.get_text() for page in doc])
            policy_texts.append(text)
    policy_text = "\n\n".join(policy_texts)
except Exception as e:
    print(f"Error reading policy document: {e}")

# ========= 🔹 Chunking the Policy Text for Retrieval 🔹 =========
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = text_splitter.split_text(policy_text)

# ========= 🔹 Vector Database (FAISS) 🔹 =========
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(texts, embeddings)

# ========= 🔹 LLM Configuration (Meta Llama 3) 🔹 =========
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B",
    temperature=0.3,
    top_p=0.3,
    task="text-generation",
    model_kwargs={"max_length": 2048}  # Increase for multi-step answers
)

# ========= 🔹 Multi-Query Retriever for Improved Search 🔹 =========
multi_query_retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=db.as_retriever())

# ========= 🔹 Memory-Enabled Conversational Retrieval 🔹 =========
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=multi_query_retriever, memory=memory, verbose=True
)

# ========= 🔹 Wrap QA Chain in a Tool (Fix for Agents) 🔹 =========
def retrieve_insurance_info(query: str) -> str:
    """Retrieve insurance-related information using FAISS + LLM."""
    return qa_chain.run({"question": query, "chat_history": []})

insurance_tool = Tool(
    name="Insurance Retrieval",
    func=retrieve_insurance_info,
    description="Use this tool to answer insurance policy questions."
)

# ========= 🔹 ReAct Agent for Step-by-Step Reasoning mem =========
agent = initialize_agent(
    tools=[insurance_tool],  # ✅ Uses the properly wrapped tool
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# ========= 🔹 API Endpoints 🔹 =========
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = agent.run(user_question)
        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# ========= 🔹 Run Flask App 🔹 =========
if __name__ == "__main__":
    app.run(debug=True)