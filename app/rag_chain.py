from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from app.config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)

# ------------------------
# Vector Store
# ------------------------
def load_vectorstore(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(docs, embeddings)

# ------------------------
# RAG Chain Factory
# ------------------------
def build_qa_chain(data_path: str):
    vectorstore = load_vectorstore(data_path)
    #retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    #retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5, "fetch_k": 10})

    
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 12
    }
)



    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=90,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template = """
You are Mr. Landon, the manager of Landon Hotel.

Use only the information in the context to answer the question.
If the answer cannot be found in the context, reply:
"I can't assist you with that, sorry!"

Context:
{context}

Question: {question}
Answer:
"""


    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
