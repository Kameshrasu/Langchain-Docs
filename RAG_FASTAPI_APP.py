from fastapi import FastAPI,File,UploadFile
import PyPDF2
from io import BytesIO
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
import uvicorn








load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode(docs):
    global vectors
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_text(docs)
    vectors = Chroma.from_texts(final_documents, embeddings)


def model_response(query):
    if vectors is None:
        return "No documents have been uploaded yet."
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": query})
    return response['answer']

@app.post("/RAG_API")
def rag_api(file: UploadFile = File(...), question: str = ""):
    try:
        file_content = file.file.read()
        file_ext = file.filename.lower().split('.')[-1]

        if file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        else:
            try:
                text = file_content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = file_content.decode("latin-1")
                except UnicodeDecodeError:
                    return JSONResponse(status_code=400, content={"error": "Unable to decode file. Please ensure it's a text or PDF file."})

        encode(text)
        answer = model_response(question)
        return {"answer": answer}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

