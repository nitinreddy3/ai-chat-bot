from flask_cors import CORS
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
CORS(app)

# os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

policy_document_path = "sompodom_merged.pdf"
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
llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B", model_kwargs={"temperature": 0.5,"max_length": 200,
    "top_p": 0.3}) #meta-llama/Meta-Llama-3-8B
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(),verbose=False)
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = qa_chain.run(user_question)
        pattern = r"Helpful Answer:(.*)"
        match = re.search(pattern, response, re.DOTALL)
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





    # "am i covered for covid 19 disease in Zurich Domestic Travel Insurance, is it an add on benefit or pre included?"
    # i don't think covid 19 cover is pre-included in zurich travel domestic policy.
    # is covid 19 pre-included in zurich travel domestic policy.
    # how many plans does Zurich international travel have?