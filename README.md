# 📚 Multi-PDF Assistant (RAG + Gradio) (Best for Vietnamese Users)

Upload 2–3 PDF documents **on the same topic**, ask a question, and the
assistant will:

1. Find the relevant pages across **all** the documents.
2. Write an answer based on those pages.
3. Tell you **which source** (file name + page number) covers that topic.

This is a simple RAG (Retrieval-Augmented Generation) pipeline: a
multilingual embedding model retrieves the relevant pages, then a small
language model reads those pages and writes the answer — no fine-tuning,
no paid API required.

## ⚙️ How it works

| Step | Model / library | Role |
|---|---|---|
| Read PDFs | `pypdf` | Extracts text per page, skips empty/scanned-image pages |
| Retrieval | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embeds the question and every page, ranks by cosine similarity |
| Answering | `Qwen/Qwen2.5-0.5B-Instruct` | Reads the most relevant pages and writes a sourced answer |
| Interface | `gradio` | Web UI for uploading PDFs and asking questions |

Every PDF page is stored together with its **file name** and **page
number**, so the assistant always states which document and page a piece
of information came from — and replies *"I couldn't find this
information in the documents."* when a question falls outside the
uploaded documents, instead of making things up.

## 🚀 Getting started

### Run locally

```bash
git clone <your-repo-url>
cd TroLyNhieuPDF
pip install -r requirements.txt
python app.py
```

Open the `http://127.0.0.1:7860` link printed in the terminal.

> It also runs without a GPU, just slower at the answer-generation step.
> To get a temporary public link for demos, change the last line of
> `app.py` to `demo.launch(share=True)`.

### Run on Google Colab

Open `notebook/TroLyNhieuPDF.ipynb` in Colab, enable a GPU
(*Runtime → Change runtime type → T4 GPU*), then run the cells top to
bottom.

## 📁 Project structure

```
.
├── app.py                       # Full logic + UI, runs standalone
├── requirements.txt             # Dependencies
├── notebook/
│   └── TroLyNhieuPDF.ipynb      # Original notebook, for Colab
└── README.md
```

## 🧪 Testing tip

Try asking something that is **not** in the documents — the assistant
should reply "I couldn't find this information in the documents."
instead of making up an answer.

## 📄 License

Released under the [MIT License](LICENSE).
