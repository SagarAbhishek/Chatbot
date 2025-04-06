from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import torch

app = Flask(__name__)
CORS(app)  # Allow frontend to make requests to this backend

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Phi-2 model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Load embedding model for retrieving context
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load knowledge base
with open("occams_contant.txt", "r") as file:
    documents = file.readlines()

# Convert documents into vector embeddings
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# Store embeddings in FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def get_relevant_context(query, num_results=1):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, num_results)
    
    relevant_docs = [documents[i] for i in indices[0]]
    return " ".join(relevant_docs)
    
def chatbot_answer(query):
    context = get_relevant_context(query)

    input_text = f"""
    You are a friendly female assistant named LiSA, part of Occams Advisory. Give the response in 20 words.
    Context: {context}
    
    Question: {query}
    """

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.2,
        top_p=0.9,
        top_k=50,
        do_sample=True 
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if 'return "' in response:
        start_index = response.find('return "') + len('return "')
        end_index = response.find('"', start_index)
        extracted_response = response[start_index:end_index]
        return extracted_response
    else:
        return "Sorry, no valid response found."
    # return response.strip()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message received"}), 400

    bot_reply = chatbot_answer(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
