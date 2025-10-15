# RAG-Ceramics

**RAG-Ceramics** is an open-source project that implements an **intelligent question-answering system** for the **ceramic domain**. The system leverages **Retrieval-Augmented Generation (RAG)**, combining **BM25 keyword retrieval** and **FAISS semantic search** to efficiently access multi-source ceramic knowledge. Large language models (LLMs) are used for **context-aware answer generation**, and a **dialog memory module** enables coherent multi-turn conversations. This project aims to support **digital preservation, knowledge dissemination, and intelligent access** to ceramic expertise.

---

## Features

- Converts PDF documents to Markdown using **Mineru**.
- Supports hybrid retrieval: **BM25 (keyword search)** + **FAISS (semantic search)**.
- Integrates **LLM-based answer generation** and **multi-turn dialogue memory**.
- Provides a **Streamlit interface** for interactive user queries.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/RAG-Ceramics.git
cd RAG-Ceramics
```


### 2. Install dependencies

Make sure Python 3.8+ is installed, then run:

<pre class="overflow-visible!" data-start="1200" data-end="1243"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt</span></span></code></div></div></pre>

### 3. Mineru installation

This project uses **Mineru** to convert PDFs to Markdown.

1. Install Mineru:

<pre class="overflow-visible!" data-start="1350" data-end="1438"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -U </span><span>"magic-pdf[full]"</span><span> -i https://pypi.tuna.tsinghua.edu.cn/simple
</span></span></code></div></div></pre>

2. Download the required models:

<pre class="overflow-visible!" data-start="1472" data-end="1509"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python download_models.py
</span></span></code></div></div></pre>

3. (Optional) Enable GPU:

   * On  **Windows** : edit `C:\Users\<username>\magic-pdf.json`
   * On  **Linux** : edit `/home/<username>/magic-pdf.json`
   * Change `"device"` to `"cuda"`

   ## Usage


   1. **Upload documents**
      * Upload **PDF files** to the `upload` folder.
      * If already in  **Markdown format** , place them in the `data` folder.
   2. **Run the application**

   <pre class="overflow-visible!" data-start="1891" data-end="1924"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>streamlit run main.py
   </span></span></code></div></div></pre>

   3. Ensure all required third-party libraries are installed before starting.


   ## License

   This project is open-source and released under the  **MIT License** .

   ---

   ## Acknowledgements

   * **Mineru** for PDF to Markdown conversion.
   * **Llama-Index** and **LangChain** for RAG framework and LLM integration.
   * **Streamlit** for the interactive frontend.
