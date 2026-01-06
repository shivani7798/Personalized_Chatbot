from flask import Flask, request, jsonify, render_template
from app.rag_chain import build_qa_chain

def create_app():
    app = Flask(__name__)

    # Build RAG chain once at startup
    qa_chain = build_qa_chain("data/website_text.txt")

    @app.route("/", methods=["GET"])
    def home():
        return render_template("index.html")

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        question = data.get("question", "").strip()

        if question.lower() in ["hi", "hello", "hey"]:
            return jsonify({
                "response": "Hello! I’m Mr. Landon, your hotel assistant. How can I help you today?"
            })

        if "renewable" in question.lower()  or "energy" in question.lower() :
            question = "Sustainability and energy use: " + question


        if question.lower() in ["rooms?", "rooms", "what rooms do you have?", "room types?"]:
            question = "What types of rooms are available at the Landon Hotel?"


        result = qa_chain.invoke({"query": question})

        # Handle both string and dict outputs safely
        if isinstance(result, dict):
            answer = result.get("result", "")
        else:
            answer = result

        return jsonify({"response": answer})

    return app
