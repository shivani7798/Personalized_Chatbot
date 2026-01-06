# Install these if not already
# pip install transformers langchain flask

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

# Load a small model for fast testing (Flan-T5 small)
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Hugging Face pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # -1 for CPU, 0 for GPU
    max_new_tokens=200,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Load your prompt text
prompt_text = open('website_text.txt', 'r').read()

# Create template
hotel_assistant_template = prompt_text + """
You are the hotel manager of Landon Hotel, named "Mr. Landon". 
You only answer questions about Landon Hotel. 
If a question is not about Landon Hotel, respond with, "I can't assist you with that, sorry!" 
Question: {question} 
Answer: 
"""

hotel_assistant_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=hotel_assistant_template
)

# Create LangChain LLMChain
llm_chain = LLMChain(prompt=hotel_assistant_prompt_template, llm=llm)

def query_llm(question):
    response = llm_chain.run({"question": question})
    return response

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    question = data["question"]
    response = query_llm(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
