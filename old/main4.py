from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vectorstore(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_vectorstore("website_text.txt")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA



prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Mr. Landon, the manager of Landon Hotel.

Answer the question using ONLY the context below.
If the answer is not found in the context, say:
"I can't assist you with that, sorry!"

Context:
{context}

Question: {question}
Answer (1-2 sentences):
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

def query_llm(question):
    # Greeting handled softly (optional)
    if question.lower().strip() in ["hi", "hello", "hey"]:
        return "Hello! I’m Mr. Landon, your hotel assistant. How can I help you today?"

    result = qa_chain.invoke({"query": question})
    return result["result"]
