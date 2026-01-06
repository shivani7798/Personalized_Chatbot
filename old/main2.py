# Install if not done
# pip install transformers langchain flask

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Load small model ---
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU, change to 0 for GPU
    max_new_tokens=150,  # keep output short
)

# --- Load website text and split into chunks ---
def load_chunks(file_path, chunk_size=500):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

website_chunks = load_chunks("website_text.txt")

# --- Query function with chunked context ---
def query_llm(question):
    # Find the chunk most relevant to the question
    # For simplicity, we just concatenate first few chunks (small memory usage)
    context = " ".join(website_chunks[:3])  # use first 3 chunks (~1500 tokens)
    prompt = f"{context}\n\nYou are the hotel manager of Landon Hotel, named 'Mr. Landon'. You only answer questions about Landon Hotel. If a question is not about Landon Hotel, respond with: 'I can't assist you with that, sorry!'\n\nQuestion: {question}\nAnswer:"
    result = pipe(prompt)
    return result[0]['generated_text'].strip()

# --- Flask app ---
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    question = data.get("question", "")
    response = query_llm(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
