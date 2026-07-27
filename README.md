# Enterprise Multi-Agent AI Assistant - Nikky Frozen

Aplikasi AI Assistant berbasis Multi-Agent System menggunakan LangChain, LangGraph, Ollama (Llama 3.2), ChromaDB, dan Streamlit. Sistem ini dirancang untuk membantu operasional bisnis frozen food (Nikky Frozen) di beberapa divisi.

---

## Fitur Utama

1. **Multi-Agent AI**: Terdiri dari 5 AI Agent (Customer Service, Inventory, Finance, Marketing, dan Manager).
2. **LangGraph Workflow**: Pengaturan jalur percakapan dinamis (routing) dari Manager Agent ke agent spesialis yang paling sesuai dengan pertanyaan user.
3. **RAG (Retrieval Augmented Generation)**: Pipeline pencarian dokumen (SOP, FAQ, Produk, Penjualan, Inventory) menggunakan ChromaDB dan HuggingFace Sentence Transformers embeddings.
4. **Real-time Logging & Monitoring**: Log aktivitas langkah demi langkah dari agent yang bekerja di balik layar.
5. **Metrik Evaluasi**: Perhitungan Accuracy, Effectiveness, Efficiency, Hallucination Risk, serta Explainability (sumber dokumen).

---

## Struktur Project

```text
enterprise-ai/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Setup and system description
├── agents/                     # Specialist Agent definitions
│   ├── base_agent.py           # Parent Agent class
│   ├── customer_agent.py       # Customer Service Specialist Agent
│   ├── inventory_agent.py      # Inventory Specialist Agent
│   ├── finance_agent.py        # Finance Specialist Agent
│   ├── marketing_agent.py      # Marketing Specialist Agent
│   └── manager_agent.py        # Supervisor/Router Agent
├── rag/                        # RAG Pipeline modules
│   ├── loader.py               # Document loader (PDF, CSV, TXT)
│   ├── splitter.py             # Recursive text splitter
│   ├── embedding.py            # Sentence Transformers Embeddings
│   ├── vector_store.py         # ChromaDB Vector Store wrapper
│   └── retriever.py            # Similarity search retriever
├── llm/
│   └── ollama_model.py         # Ollama LLM setup
├── workflows/
│   └── graph.py                # LangGraph state & routing graph
├── documents/                  # SOP & FAQ document storage
├── database/
│   └── chroma_db/              # Persistent ChromaDB vector database
└── utils/
    ├── config.py               # Paths and model configuration
    └── evaluator.py            # Accuracy & Hallucination evaluation metrics
```

---

## Cara Instalasi & Menjalankan Project

### 1. Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal:
* Python 3.11+
* Ollama

### 2. Instalasi & Menjalankan Ollama
1. Unduh dan pasang Ollama dari situs resminya: [https://ollama.com](https://ollama.com).
2. Jalankan Ollama di perangkat Anda.
3. Buka terminal/cmd dan jalankan perintah berikut untuk mengunduh model **Llama 3.2**:
   ```bash
   ollama pull llama3.2
   ```

### 3. Instalasi Dependency Python
1. Buka folder `enterprise-ai` di terminal.
2. Buat Virtual Environment (opsional namun direkomendasikan):
   ```bash
   python -m venv venv
   # Aktifkan venv di Windows:
   .\venv\Scripts\activate
   # Atau macOS/Linux:
   source venv/bin/activate
   ```
3. Pasang library yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Menjalankan Aplikasi
Jalankan dashboard Streamlit dengan perintah:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

---

## Alur Kerja Multi-Agent (LangGraph)

1. **User Input**: Pengguna mengirimkan pertanyaan melalui antarmuka chat Streamlit.
2. **Manager (Supervisor)**: Menerima query lalu menggunakan Ollama untuk menentukan kategori divisi (`customer_service`, `inventory`, `finance`, `marketing`, atau `general`).
3. **Specialist Agent**:
   * **Customer Service Agent**: Mengambil data dari `SOP.txt` dan `faq.txt` jika berhubungan dengan pertanyaan umum.
   * **Inventory Agent**: Mengecek stok barang di `produk.csv` dan merekomendasikan restock produk yang stoknya di bawah 10 unit.
   * **Finance Agent**: Membaca transaksi di `penjualan.csv` untuk mencocokkan status invoice.
   * **Marketing Agent**: Menganalisis produk terlaris di `penjualan.csv` dan mengusulkan program diskon.
4. **Aggregator Node**: Manager mengompilasi jawaban dari specialist agent tersebut menjadi satu jawaban final terstruktur untuk dikirimkan kembali ke pengguna.
