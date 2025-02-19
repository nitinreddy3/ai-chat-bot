from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
import os
import fitz  # PyMuPDF
import re
app = Flask(__name__)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
policy_document_path = "./travel.pdf"  # Example: "travel_policy.pdf"
try:
    with fitz.open(policy_document_path) as doc:
        policy_text = "\n".join([page.get_text() for page in doc])
    print("Policy document",policy_text)
except FileNotFoundError:
    print(f"Error: Policy document not found at {policy_document_path}")
except Exception as e:
    print(f"Error reading policy document: {e}")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # Adjust chunk size as needed
texts = text_splitter.split_text(policy_text)
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(texts, embeddings)
# llm = OpenAI(temperature=0)
llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B", model_kwargs={"temperature": 0.2,"max_length": 2,
    "top_p": 0.2})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(),verbose=False)
# @app.route('/api/query', methods=['POST'])
# def handle_query():
#     user_query = request.json.get('query')
#     if not user_query:
#         return jsonify({'error': 'No query provided'}), 400
#     try:
#         result = qa_chain.run(user_query)
#         return jsonify({'response': result})
#     except Exception as e:
#         print(f"Error processing query: {e}") # Log the error for debugging
#         return jsonify({'error': str(e)}), 500  # Return error to the frontend
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    try:
        response = qa_chain.run(user_question)
        pattern = r"Helpful Answer:(.*)"
        match = re.search(pattern, response, re.DOTALL)
        # Check if a match was found and extract the text
        if match:
            helpful_answer = match.group(1).strip()
            print("Helpful Answer and Text After It:")
            print(helpful_answer)
            return jsonify({"answer": helpful_answer})
        else:
            print("No helpful answer found.")
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
if __name__ == "__main__" :
    app.run(debug=True)