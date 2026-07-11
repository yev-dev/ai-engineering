import os
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from `\.models\..*",
)

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import faiss
import pymupdf


# Ensure the vector_db folder exists
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)


def _load_and_convert_with_pymupdf(file_path):
    document = pymupdf.open(file_path)
    sections = []
    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()
            sections.append(f"# Page {page_number}\n\n{text}")
    finally:
        document.close()

    return "\n\n".join(section for section in sections if section.strip())


# Load and convert PDF to markdown content
def load_and_convert_document(file_path):
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as exc:
        warnings.warn(
            f"Falling back to PyMuPDF text extraction because docling conversion failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _load_and_convert_with_pymupdf(file_path)


# Split markdown into chunks
def get_markdown_splits(markdown_content):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return splitter.split_text(markdown_content)


# Create or load the vector store
def create_or_load_vector_store(filename, chunks, embeddings):
    vector_db_path = Path(VECTOR_DB_FOLDER) / f"{filename}.faiss"

    if vector_db_path.exists():
        vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        single_vector = embeddings.embed_query("initialize")
        index = faiss.IndexFlatL2(len(single_vector))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(chunks)
        vector_store.save_local(str(vector_db_path))
    return vector_store


# Build RAG chain
def build_rag_chain(
    retriever,
    model_name="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    response_type="Plain Text",
):
    prompt = """
        You are an assistant for financial data analysis. Use the retrieved context to answer questions.
        If you don't know the answer, say so.

        Response format requirement: {response_type}
        Formatting rules:
        - Plain Text: respond with plain text only. Do not use markdown lists, headings, or code fences.
        - Markdown: respond using clear markdown formatting.
        - Python Code: respond with Python code only inside a single fenced python code block, with no extra explanation outside the code block.

        Question: {question}
        Context: {context}
        Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(prompt)
    model = ChatOllama(model=model_name, base_url=base_url)
    return (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
            "response_type": lambda _: response_type,
        }
        | prompt_template
        | model
        | StrOutputParser()
    )
