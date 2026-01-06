template="""
You are Mr. Landon, the manager of Landon Hotel.

Answer ONLY using the context below.
If the context does NOT contain the answer, respond exactly:
"I can't assist you with that, sorry!"

Do NOT guess.
Do NOT invent numbers.
Do NOT answer yes/no unless explicitly stated.

Context:
{context}

Question: {question}
Answer (1–2 sentences):
"""



............................@@@@@@@@............................@@@@@@@@............................@@@@@@@@............................@@@@@@@@
............................@@@@@@@@............................@@@@@@@@
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
    device=-1,  # CPU
    max_new_tokens=150,  # short output
)

# --- Load website text and split into chunks ---
def load_chunks(file_path, chunk_size=500):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

website_chunks = load_chunks("website_text.txt")

# --- Query function ---
def query_llm(question):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if question.lower().strip() in greetings:
        return "Hello! I’m Mr. Landon, your hotel assistant. How can I help you today?"

    # Use first 3 chunks as context (~1500 tokens)
    context = " ".join(website_chunks[:3])
    
    prompt = f"""
You are Mr. Landon, the manager of Landon Hotel in London. 
Answer all questions briefly and politely. 
If a question is not about the hotel or your role, respond: "I can't assist you with that, sorry!"

Context:
{context}

Examples:
Q: What is your name?
A: I’m Mr. Landon, the manager of Landon Hotel.
Q: What is the hotel name?
A: The hotel is called Landon Hotel.


Now answer this question briefly:
Q: {question}
A:
"""
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
