# streamlit run app.py

import json
import os
import warnings
from pathlib import Path
from time import perf_counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Suppress a known upstream transformers warning emitted by transitive deps.
warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from `.models.vilt.image_processing_vilt`.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Behavior may be different and this alias will be removed in future versions\..*",
)

import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import FAISS

from pdf_utils import render_pdf_pages
from rag import (
    build_rag_chain,
    create_or_load_vector_store,
    get_markdown_splits,
    load_and_convert_document,
)
from utils import (
    append_question_history,
    clear_question_history,
    execute_python_code,
    extract_python_code,
    get_embeddings,
    load_local_model_options,
    load_question_history,
    sanitize_generated_python_code,
)

# Path to vector DB folder
VECTOR_DB_FOLDER = "vector_db"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "deepseek-r1:1.5b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
QUESTION_HISTORY_DIR = Path(VECTOR_DB_FOLDER) / "question_history"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)
os.makedirs(QUESTION_HISTORY_DIR, exist_ok=True)


def render_response_output(response_text: str, response_type: str, panel_key: str) -> None:
    if response_type == "Plain Text":
        st.text(response_text)
        return

    if response_type == "Python Code":
        raw_code = extract_python_code(response_text)
        code = sanitize_generated_python_code(raw_code)
        if code != raw_code:
            st.caption("Auto-cleaned malformed Python syntax from model output.")
        st.code(code, language="python")

        with st.expander("Run in Streamlit (Pyodide/WebAssembly)", expanded=False):
            render_pyodide_runner(code, panel_key=panel_key)

        confirmation_key = f"confirm_python_{panel_key}"
        confirmed = st.checkbox(
            "I understand this will execute model-generated Python locally.",
            key=confirmation_key,
        )
        if st.button("Execute Python", key=f"execute_python_{panel_key}"):
            if not confirmed:
                st.warning("Confirm execution before running Python code.")
                return
            stdout_text, stderr_text = execute_python_code(code)
            if stdout_text.strip():
                st.text(stdout_text)
            if stderr_text.strip():
                st.error(stderr_text)
            elif not stdout_text.strip():
                st.success("Python code executed successfully with no output.")
        return

    st.markdown(response_text.replace('$', '\\$'))


def render_pyodide_runner(initial_code: str, panel_key: str) -> None:
        code_json = json.dumps(initial_code)
        html = f"""
<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, monospace; border: 1px solid #ddd; border-radius: 8px; padding: 10px;\">
    <div style=\"font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 14px; margin-bottom: 8px;\">
        A WebAssembly-powered Python kernel backed by Pyodide
    </div>
    <textarea id=\"code_{panel_key}\" style=\"width: 100%; height: 170px;\"></textarea>
    <div style=\"margin-top: 8px; display: flex; gap: 8px; align-items: center;\">
        <button id=\"run_{panel_key}\">Run In Browser</button>
        <span id=\"status_{panel_key}\" style=\"font-family: system-ui, -apple-system, Segoe UI, sans-serif; font-size: 12px; color: #444;\"></span>
    </div>
    <pre id=\"output_{panel_key}\" style=\"white-space: pre-wrap; margin-top: 10px; background: #f7f7f7; padding: 10px; border-radius: 6px; max-height: 180px; overflow: auto;\"></pre>
</div>
<script>
    const initialCode = {code_json};
    const codeEl = document.getElementById("code_{panel_key}");
    const runEl = document.getElementById("run_{panel_key}");
    const statusEl = document.getElementById("status_{panel_key}");
    const outputEl = document.getElementById("output_{panel_key}");
    codeEl.value = initialCode;

    async function ensurePyodide() {{
        if (!window.__streamlitPyodideReady) {{
            statusEl.textContent = "Loading Pyodide runtime...";
            if (!window.loadPyodide) {{
                await new Promise((resolve, reject) => {{
                    const script = document.createElement("script");
                    script.src = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js";
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                }});
            }}
            window.__streamlitPyodideReady = await window.loadPyodide();
        }}
        return window.__streamlitPyodideReady;
    }}

    runEl.onclick = async () => {{
        runEl.disabled = true;
        outputEl.textContent = "";
        try {{
            const pyodide = await ensurePyodide();
            statusEl.textContent = "Running...";

            const stdout = [];
            const stderr = [];
            pyodide.setStdout({{ batched: (msg) => stdout.push(msg) }});
            pyodide.setStderr({{ batched: (msg) => stderr.push(msg) }});

            const result = await pyodide.runPythonAsync(codeEl.value);
            const chunks = [];
            if (stdout.length) chunks.push(stdout.join("\n"));
            if (result !== undefined) chunks.push(String(result));
            if (stderr.length) chunks.push("\n[stderr]\n" + stderr.join("\n"));
            outputEl.textContent = chunks.join("\n\n") || "(no output)";
            statusEl.textContent = "Done";
        }} catch (err) {{
            outputEl.textContent = String(err);
            statusEl.textContent = "Failed";
        }} finally {{
            runEl.disabled = false;
        }}
    }};
</script>
"""
        components.html(html, height=460, scrolling=True)


@st.cache_data(ttl=10)
def get_local_model_options() -> tuple[list[str], list[str], str | None]:
    return load_local_model_options(
        base_url=OLLAMA_BASE_URL,
        default_chat_model=DEFAULT_CHAT_MODEL,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
    )

# Function to display PDF content as images in the sidebar
def display_pdf_in_sidebar(pdf_path, file_name):
    try:
        images_folder = Path(VECTOR_DB_FOLDER) / file_name / "images"
        image_paths = render_pdf_pages(pdf_path, images_folder, zoom=1.5)
        for page_index, img_path in enumerate(image_paths, start=1):
            st.sidebar.image(str(img_path), caption=f"Page {page_index}", width="stretch")
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {str(e)}")

# Streamlit title and layout
st.title("Financial Data Analysis")

available_chat_models, available_embedding_models, model_load_error = get_local_model_options()
default_model_index = (
    available_chat_models.index(DEFAULT_CHAT_MODEL)
    if DEFAULT_CHAT_MODEL in available_chat_models
    else 0
)
selected_model = st.selectbox(
    "Select Local Ollama Model",
    available_chat_models,
    index=default_model_index,
)

default_embedding_index = (
    available_embedding_models.index(DEFAULT_EMBEDDING_MODEL)
    if DEFAULT_EMBEDDING_MODEL in available_embedding_models
    else 0
)
selected_embedding_model = st.selectbox(
    "Select Local Embedding Model",
    available_embedding_models,
    index=default_embedding_index,
)

response_type = st.selectbox(
    "Select Response Type",
    ["Plain Text", "Markdown", "Python Code"],
    index=1,
)

if model_load_error:
    st.warning(model_load_error)

st.caption(
    "Use the same embedding model that was used when the vector DB was created."
)

# Dropdown to select vector DB or upload a new document
vector_db_options = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]
vector_db_options.append("Upload New Document")  # Add option to upload a new document
selected_vector_db = st.selectbox("Select Vector DB or Upload New Document", vector_db_options, index=0)

history_vector_db = selected_vector_db if selected_vector_db != "Upload New Document" else "__upload__"
if st.session_state.get("history_vector_db") != history_vector_db:
    st.session_state["history_vector_db"] = history_vector_db
    st.session_state["question_history"] = load_question_history(history_vector_db, QUESTION_HISTORY_DIR)

with st.expander("Previous Questions", expanded=False):
    history = st.session_state.get("question_history", [])
    if selected_vector_db == "Upload New Document":
        st.caption("Question history is available after you select an existing vector DB.")
    elif not history:
        st.caption("No saved question history yet for this vector DB.")
    else:
        if st.button("Clear History", key=f"clear_history_{history_vector_db}"):
            clear_question_history(history_vector_db, QUESTION_HISTORY_DIR)
            st.session_state["question_history"] = []
            st.rerun()

        for index, item in enumerate(history[:10], start=1):
            st.markdown(f"**Q:** {item['question']}")
            st.caption(
                f"Model: {item['chat_model']} | Embedding: {item['embedding_model']} | "
                f"Type: {item.get('response_type', 'Markdown')} | Time: {item['answer_seconds']:.2f}s"
            )
            if item.get("answer"):
                render_response_output(
                    item["answer"],
                    item.get("response_type", "Markdown"),
                    panel_key=f"history_{history_vector_db}_{index}",
                )
            if st.button("Reuse Question", key=f"reuse_question_{history_vector_db}_{index}"):
                st.session_state["question_input"] = item["question"]
                st.rerun()

# If 'Upload New Document' is selected, show the file uploader
if selected_vector_db == "Upload New Document":
    uploaded_file = st.file_uploader("Upload a PDF file for analysis", type=["pdf"])

    # Process the uploaded PDF
    if uploaded_file:
        st.sidebar.subheader("Uploaded PDF")
        st.sidebar.write(uploaded_file.name)

        # Save the PDF file temporarily and display it
        temp_path = f"temp_{uploaded_file.name}"
        document_binary = uploaded_file.read()
        with open(temp_path, "wb") as f:
            f.write(document_binary)

        # Display PDF in the sidebar (show all pages)
        display_pdf_in_sidebar(temp_path, uploaded_file.name.split('.')[0])

        # PDF processing button
        if st.button("Process PDF and Store in Vector DB"):
            with st.spinner("Processing document..."):
                start_time = perf_counter()

                # Convert PDF to markdown directly
                markdown_content = load_and_convert_document(temp_path)
                chunks = get_markdown_splits(markdown_content)

                # Initialize embeddings
                embeddings = get_embeddings(selected_embedding_model, OLLAMA_BASE_URL)

                # Create or load vector DB and store PDF along with it
                vector_store = create_or_load_vector_store(uploaded_file.name.split(".")[0], chunks, embeddings)

                # Ensure vector DB and PDF are stored correctly
                vector_db_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name.split('.')[0]}.faiss"
                vector_store.save_local(str(vector_db_path))  # Save FAISS vector store

                # Store the PDF file alongside the vector DB
                pdf_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(document_binary)

                st.success("PDF processed and stored in the vector database.")
                st.caption(
                    f"Document processing completed in {perf_counter() - start_time:.2f} seconds."
                )

                # Clean up the temporary file
                Path(temp_path).unlink()

elif selected_vector_db != "Upload New Document":
    # Load the selected vector DB
    vector_db_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.faiss"
    if vector_db_path.exists():
        embeddings = get_embeddings(selected_embedding_model, OLLAMA_BASE_URL)
        vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

        # Display PDF in the sidebar
        pdf_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.pdf"
        if pdf_path.exists():
            display_pdf_in_sidebar(pdf_path, selected_vector_db)
        else:
            st.sidebar.warning("PDF file not found for the selected vector DB.")
    else:
        st.sidebar.warning(f"Vector DB '{selected_vector_db}' not found.")

# Question input section
question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the company's revenue for the quarter?",
    key="question_input",
)

submit_clicked = st.button("Submit Question")

latest_response = st.session_state.get("latest_response")
if latest_response and not submit_clicked:
    st.subheader("Latest Response")
    render_response_output(
        latest_response["answer"],
        latest_response.get("response_type", "Markdown"),
        panel_key="latest_response",
    )

# Button to process and generate answers
if submit_clicked and question and selected_vector_db != "Upload New Document":
    with st.spinner("Answering your question..."):
        start_time = perf_counter()

        # Build retriever from the selected vector store
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5})

        # Build and run the RAG chain
        rag_chain = build_rag_chain(
            retriever,
            selected_model,
            OLLAMA_BASE_URL,
            response_type,
        )

        # Create a placeholder for streaming response
        response_placeholder = st.empty()  # Create an empty placeholder for the answer

        # Stream the response as it is generated
        response = ""
        for chunk in rag_chain.stream(question):
            response += chunk  # Append each chunk of the response
            if response_type == "Plain Text":
                response_placeholder.text(response)
            else:
                response_placeholder.markdown(response.replace('$', '\\$'))

        response_placeholder.empty()
        st.session_state["latest_response"] = {
            "question": question,
            "answer": response,
            "response_type": response_type,
        }

        st.subheader("Latest Response")
        render_response_output(response, response_type, panel_key="latest_response")

        st.caption(
            f"Answer generated in {perf_counter() - start_time:.2f} seconds."
        )

        history_entry = {
            "question": question,
            "answer": response,
            "vector_db": selected_vector_db,
            "chat_model": selected_model,
            "embedding_model": selected_embedding_model,
            "response_type": response_type,
            "answer_seconds": perf_counter() - start_time,
        }
        append_question_history(selected_vector_db, QUESTION_HISTORY_DIR, history_entry)
        st.session_state["question_history"] = load_question_history(selected_vector_db, QUESTION_HISTORY_DIR)

