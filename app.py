import os
import requests
import fitz  # PyMuPDF
import re
from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, OpenAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from flask_cors import CORS
from tavily import TavilyClient
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)
CORS(app)

tavily = TavilyClient(api_key=os.getenv("TAVILY_KEY"))
policy_text = ""

POLICY_DOCUMENT_PATH = ["sompodom_merged.pdf", "Zurich_APAC.pdf","SOMPO_Domestic.pdf"]
print("log1")
try:
    policy_texts = []
    for file_path in POLICY_DOCUMENT_PATH:
        with fitz.open(file_path) as doc:
            text = "\n".join([page.get_text() for page in doc])
            policy_texts.append(text)
            # print(f"Policy document ({file_path}):", text)

    # Combine texts if needed
    policy_text = "\n\n".join(policy_texts)
    print(policy_text)
except FileNotFoundError:
    print(f"Error: Policy document not found at {POLICY_DOCUMENT_PATH}")
except Exception as e:
    print(f"Error reading policy document: {e}")

print("log2")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)  # Adjust chunk size as needed
texts = text_splitter.split_text(policy_text)


print("log3")
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(texts, embeddings)

print("log4")
# llm = OpenAI(temperature=0)
# llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B", model_kwargs={"temperature": 0.5,"max_length": 200,
#     "top_p": 0.3}) #meta-llama/Meta-Llama-3-8B

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B",
    temperature=0.5,  # Move outside model_kwargs
    top_p=0.3,        # Move outside model_kwargs
    task= "text-generation",  # Move outside model_kwargs
    model_kwargs={"max_length": 200}  # Keep only max_length inside
)

print("log5")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(),verbose=False)
print("log6")
# ========== API ROUTES ==========0

# @app.route("/ask", methods=["POST"])
# def ask():
#     user_question = request.json.get("question")
#     if not user_question:
#         return jsonify({"error": "No question provided"}), 400
#     # response = general_online_queries(user_question)
#     # return jsonify(response)
#     try:
#         print("log7")
#         print("Checking Hugging Face API key:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
#         response = qa_chain.run(user_question)
#         print("log8")
#         pattern = r"Helpful Answer:(.*)"
#         match = re.search(pattern, response, re.DOTALL)
#         if match:
#             helpful_answer = match.group(1).strip()
#             print("Helpful Answer and Text After It:")
#             print(helpful_answer)
#             return jsonify({"answer": helpful_answer})
#         else:
#             print("No helpful answer found.")

#     except Exception as e:
#         import traceback
#         print("Full error traceback:", traceback.format_exc())
#         return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        print("log7")
        response = qa_chain.run(user_question)
        return jsonify(response)
        print("log8")

        pattern = r"Helpful Answer:(.*)"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            helpful_answer = match.group(1).strip()
            print("Helpful Answer and Text After It:")
            print(helpful_answer)
            return jsonify({"answer": helpful_answer})
        else:
            print("No helpful answer found.")
            return jsonify({"answer": "Sorry, I couldn't find a relevant response."})  # Return fallback response

    except Exception as e:
        import traceback
        print("Full error traceback:", traceback.format_exc())  # Print full error details
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/recommend/plan", methods=["POST"])
def recommend():
    template = """
    You are an AI travel insurance assistant. Based on the latest travel insights:

    - {tavily_summary}

    Provide a personalized insurance recommendation for someone visiting {destination}.
    Highlight important add-ons based on current risks.
    """
    travel_data = fetch_travel_data("France")
    prompt = PromptTemplate(input_variables=["destination", "tavily_summary"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    recommendation = chain.run(destination="France", tavily_summary=travel_data["answer"])
    print("AI Insurance Recommendation:\n", recommendation)


# ========== HELPER FUNCTIONS ==========
def generate_recommendation(user_question, destination):
    # Get coverage info
    coverage_info = qa_chain.run(user_question)
    # Get live events for destination
    events =fetch_travel_data(destination)
    # Craft a combined prompt for the LLM:
    combined_prompt = (
        f"Based on the following policy details:\n{coverage_info}\n\n"
        f"and considering these recent events in {destination}:\n{', '.join(events)}\n\n"
        "What are the best recommendations for insurance coverage and add-ons? "
        "Answer concisely."
    )
    recommendation = llm(combined_prompt)
    return recommendation

def fetch_travel_data(destination, start_date, end_date):
    query = f"Upcoming events, holidays, health alerts, and travel risks in {destination} for duration {start_date} to {end_date} or in general."
    results = tavily.search(query=query, search_depth="advanced", include_answer=True)

    # Extract useful information
    return {
        "answer": results.get("answer", "No AI summary available"),
        "search_results": [{"title": r["title"], "url": r["url"]} for r in results.get("results", [])]
    }

def general_online_queries(query):
    results = results = tavily.search(query=query, search_depth="advanced", include_answer=True) or {} #tavily.search(query=query, search_depth="advanced", include_answer=True)
    return {
        "answer": results.get("answer", "No AI summary available"),
        "search_results": [
            {"title": r["title"], "url": r["url"]}
            for r in results.get("results", [])
        ] or [{"title": "No relevant results found", "url": ""}]
    }

if __name__ == "__main__" :
    app.run(debug=True)