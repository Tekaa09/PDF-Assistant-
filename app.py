"""
Multi-PDF Assistant (RAG + Gradio)

Upload 2-3 PDF documents on the same topic, ask a question, and the
assistant will:
1. Find the relevant pages across ALL documents.
2. Write an answer based on those pages.
3. State which source (file name + page number) covers that topic.

Run locally:
    pip install -r requirements.txt
    python app_v2.py
    -> open the http://127.0.0.1:7860 link printed in the terminal

For a temporary public link (e.g. for a demo), change the last line to
demo.launch(share=True).

Works without a GPU too, just slower at the answer-generation step.
"""

import os
import re

import torch
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# PART 1 - LOAD THE AI MODELS
print("Loading AI models... (first run can take 2-3 minutes)")

# Retriever: turns text into embeddings to find relevant pages
retriever_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Generator: reads the retrieved pages and writes the answer
generator_model = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)

print("AI ready! Device:", "GPU" if torch.cuda.is_available() else "CPU (will be slower)")

# PART 2 - CORE RAG LOGIC
# DOCUMENT_STORE holds everything the assistant "remembers". The three
# lists always stay the same length and order: content[i] is the text of
# page i, and file_name[i] / page_number[i] are its source.
DOCUMENT_STORE = {"content": [], "file_name": [], "page_number": [], "embeddings": None}

MAX_CHARS_PER_PAGE = 1000     # truncate very long pages so inference stays fast
MIN_CHARS_PER_PAGE = 30       # pages shorter than this are treated as blank
RELEVANCE_THRESHOLD = 0.30    # below this score, a page is considered unrelated
MENTION_THRESHOLD = 0.45      # at/above this score, a document clearly mentions the topic


def read_pdf(file_path):
    """Read one PDF -> (file_name, [(page_number, text), ...], error_message)."""
    file_name = os.path.basename(str(file_path))
    pages = []
    try:
        pdf = PdfReader(file_path)
        if pdf.is_encrypted:                     # handle password-protected PDFs
            try:
                pdf.decrypt("")
            except Exception:
                return file_name, [], "PDF is password-protected"
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excess blank lines
            if len(text) >= MIN_CHARS_PER_PAGE:      # skip blank pages / image-only covers
                pages.append((i + 1, text))
    except Exception as error:                   # file is corrupted or not a real PDF
        return file_name, [], f"Could not read file ({error})"

    if not pages:                                # scanned PDF with no text layer
        return file_name, [], "No text layer found (likely a scanned image) -> needs OCR"
    return file_name, pages, ""


def process_documents(file_list):
    """Load all uploaded PDFs into DOCUMENT_STORE and embed every page."""
    global DOCUMENT_STORE
    DOCUMENT_STORE = {"content": [], "file_name": [], "page_number": [], "embeddings": None}

    if not file_list:
        return "Please select 2-3 PDF files and try again.", []

    status_rows = []
    for file_path in file_list:
        file_name = os.path.basename(str(file_path))
        if not file_name.lower().endswith(".pdf"):
            status_rows.append([file_name, 0, "Not a PDF file"])
            continue

        file_name, pages, error = read_pdf(file_path)
        if error:
            status_rows.append([file_name, 0, error])
            continue

        for page_number, text in pages:          # remember each page with its source
            DOCUMENT_STORE["content"].append(text[:MAX_CHARS_PER_PAGE])
            DOCUMENT_STORE["file_name"].append(file_name)
            DOCUMENT_STORE["page_number"].append(page_number)
        status_rows.append([file_name, len(pages), "Loaded"])

    if not DOCUMENT_STORE["content"]:
        return "Could not extract any text from the uploaded files.", status_rows

    # Embed every page from every document into the same "meaning space"
    DOCUMENT_STORE["embeddings"] = retriever_model.encode(
        DOCUMENT_STORE["content"], convert_to_tensor=True, show_progress_bar=False
    )
    num_documents = len(set(DOCUMENT_STORE["file_name"]))
    return (
        f"Loaded {len(DOCUMENT_STORE['content'])} pages from {num_documents} document(s). "
        f"You can ask a question now.",
        status_rows,
    )


def mention_level(score):
    """Turn a similarity score into a human-readable label."""
    if score >= MENTION_THRESHOLD:
        return "Clearly mentioned"
    if score >= RELEVANCE_THRESHOLD:
        return "Somewhat related"
    return "Not mentioned"


def answer_question(question, top_k, answer_length):
    """Retrieve relevant pages across documents, then let the AI write an answer."""
    if DOCUMENT_STORE["embeddings"] is None:
        return "No documents loaded yet. Upload PDFs and click **Process documents** first.", [], []
    if not question or not question.strip():
        return "Please enter a question.", [], []

    question = question.strip()
    top_k = int(top_k)

    # Embed the question and compare it against every page
    question_embedding = retriever_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, DOCUMENT_STORE["embeddings"])[0].cpu().tolist()

    # Best-scoring page per document -> "which source mentions this?" table
    best_per_doc = {}
    for i, score in enumerate(scores):
        file_name = DOCUMENT_STORE["file_name"][i]
        if file_name not in best_per_doc or score > best_per_doc[file_name][1]:
            best_per_doc[file_name] = (i, score)
    ranked = sorted(best_per_doc.items(), key=lambda x: -x[1][1])
    mention_table = [
        [name, DOCUMENT_STORE["page_number"][i], round(score, 3), mention_level(score)]
        for name, (i, score) in ranked
    ]

    # Select pages for context: one best page per document first (so the
    # answer isn't skewed toward a single source), then fill remaining
    # slots with the globally top-scoring pages.
    selected = []
    for name, (i, score) in ranked:
        if score >= RELEVANCE_THRESHOLD and len(selected) < top_k:
            selected.append(i)
    global_order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for i in global_order:
        if len(selected) >= top_k:
            break
        if i not in selected and scores[i] >= RELEVANCE_THRESHOLD:
            selected.append(i)
    if not selected:                             # question is outside the documents
        selected = global_order[:1]              # still give 1 page so the AI can say "not found"
    selected.sort(key=lambda i: -scores[i])

    # Build the context text, labeling each page with its source
    context_text = ""
    source_table = []
    for i in selected:
        context_text += (
            f"\n[Source: {DOCUMENT_STORE['file_name'][i]} - Page {DOCUMENT_STORE['page_number'][i]}]\n"
            f"{DOCUMENT_STORE['content'][i]}\n"
        )
        source_table.append(
            [DOCUMENT_STORE["file_name"][i], DOCUMENT_STORE["page_number"][i], round(scores[i], 3)]
        )

    # System prompt enforces answering only from the retrieved context,
    # citing file name + page number, and admitting when info is missing.
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that reads multiple documents. Answer ONLY using the "
                       "DOCUMENTS section below. Be concise. Always cite the FILE NAME and PAGE NUMBER "
                       "for each point. If several documents cover the topic, list all of them and note "
                       "similarities/differences. If the documents don't contain the answer, reply exactly: "
                       "'I could not find this information in the documents.'",
        },
        {"role": "user", "content": f"DOCUMENTS:\n{context_text}\n\nQUESTION: {question}"},
    ]

    try:
        result = generator_model(messages, max_new_tokens=int(answer_length), do_sample=False)
        output = result[0]["generated_text"]
        answer_text = output[-1]["content"] if isinstance(output, list) else str(output)
    except Exception as error:
        return f"Error while generating the answer: {error}", source_table, mention_table

    source_list = "\n".join(
        f"- {name} - page {page} (similarity {score})" for name, page, score in source_table
    )
    final_answer = f"### Answer\n{answer_text.strip()}\n\n### Pages used\n{source_list}"
    return final_answer, source_table, mention_table


# PART 3 - GRADIO INTERFACE
def build_interface():
    with gr.Blocks(title="Multi-PDF Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 📚 Multi-PDF Assistant\n"
            "Upload **2-3 PDF documents on the same topic**, ask a question, and the assistant "
            "will answer while stating **which source and page** covers it."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ Document store")
                file_input = gr.File(
                    label="Select 2-3 PDF files (with selectable text)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                process_button = gr.Button("📥 Process documents", variant="primary")
                status_output = gr.Markdown()
                documents_table = gr.Dataframe(
                    headers=["Document", "Pages with text", "Status"],
                    label="Loading results",
                    interactive=False,
                    wrap=True,
                )

            with gr.Column(scale=2):
                gr.Markdown("### 2️⃣ Ask a question")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="E.g. Which source explains what to do if a password is leaked?",
                    lines=2,
                )
                with gr.Row():
                    topk_slider = gr.Slider(1, 8, value=4, step=1, label="Pages to retrieve (top_k)")
                    length_slider = gr.Slider(50, 400, value=180, step=10, label="Answer length (tokens)")
                ask_button = gr.Button("🤖 Ask", variant="primary")

                answer_output = gr.Markdown()
                mention_table_output = gr.Dataframe(
                    headers=["Document", "Most relevant page", "Similarity", "Mention level"],
                    label="🔎 Which source mentions this?",
                    interactive=False,
                    wrap=True,
                )
                source_table_output = gr.Dataframe(
                    headers=["Document", "Page", "Similarity"],
                    label="📄 Pages the AI read to answer",
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown(
            "💡 *Testing tip:* ask something **not** in the documents — the assistant should "
            "reply \"I could not find this information in the documents.\" instead of making it up."
        )

        process_button.click(fn=process_documents, inputs=[file_input], outputs=[status_output, documents_table])
        ask_button.click(
            fn=answer_question,
            inputs=[question_input, topk_slider, length_slider],
            outputs=[answer_output, source_table_output, mention_table_output],
        )
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, topk_slider, length_slider],
            outputs=[answer_output, source_table_output, mention_table_output],
        )

    return app


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(share=False, debug=False)

