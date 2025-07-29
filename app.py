from flask import Flask, request, render_template, session
import requests
import os
import json
# from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from flask_session import Session

# Load .env
# load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# ---- Azure Search Function ----------
def get_first_search_answer_rest(endpoint, index_name, api_key, query):
    url = f"{endpoint}/indexes/{index_name}/docs/search?api-version=2025-05-01-preview"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    body = {
        "search": query,
        "count": True,
        "vectorQueries": [
            {
                "kind": "text",
                "text": query,
                "fields": "text_vector"
            }
        ],
        "queryType": "semantic",
        "semanticConfiguration": "rag-saurabhs-semantic-configuration",
        "captions": "extractive",
        "answers": "extractive|count-3"
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 200:
        data = response.json()
        first_doc = data.get("value", [])[0] if data.get("value") else None
        if first_doc and "chunk" in first_doc:
            return first_doc["chunk"]
        else:
            return "No 'chunk' field found in first result."
    else:
        return f"Error: {response.status_code} - {response.text}"

# ---- GPT-4o Call ----
def query_openAI(context: str, user_query: str) -> str:
    endpoint = os.environ.get("AZUREAI_ENDPOINT")
    deployment = os.environ.get("AZUREAI_DEPLOYMENT")
    subscription_key = os.environ.get("AZUREAI_ENDPOINT_KEY")
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    response = client.chat.completions.create(
        model=deployment,
        max_completion_tokens=100000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful greeting assistant. "
                    "Only answer the user’s question based strictly on the provided context. take the chunk keep the same tone as in chunk and greet everyone according to it also add some unique emojis"
                    
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_query}"
            }
        ]
    )

    return response.choices[0].message.content

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None

    if request.method == "POST":
        query = request.form.get("query")

        endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        index_name = os.environ.get("AZURE_SEARCH_INDEX")
        api_key = os.environ.get("AZURE_SEARCH_KEY")

        context = get_first_search_answer_rest(endpoint, index_name, api_key, query)
        answer = query_openAI(context, query)

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
