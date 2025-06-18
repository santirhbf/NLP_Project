# European Civil Code Retrieval-Augmented Generation (RAG) System

This project provides a streamlined, multilingual, country-aware RAG system for European civil law documents. It extracts content from official Civil Codes of five countries — **Spain**, **France**, **Portugal**, **Germany**, and **Italy** — and enables advanced question-answering and translation workflows through vector database retrieval and LLM integration.

---

## Project Structure

```
.
├── main.py                          # Entry point for processing & vectorstore creation
├── config.py                        # Global configuration (paths, chunk sizes, embedding models, etc.)
├── streamlit_app.py                 # Streamlit-based web interface to interact with the RAG system
├── document_parser.py               # Utilities for extracting and cleaning text from PDF civil codes
├── rag_engine.py                    # Core logic for retrieval-augmented generation using FAISS and Gemini
├── translator.py                    # Translation and language detection using Deep Translator and LangDetect
├── country_classifier.py            # Lightweight classifier to infer the most relevant country for a query
├── Civil Codes/                     # Directory containing official civil code PDFs for each country
│   ├── Civil Code - Spain.pdf
│   ├── Civil Code - France.pdf
│   ├── Civil Code - Portugal.pdf
│   ├── Civil Code - Germany.pdf
│   └── Civil Code - Italy.pdf
├── vector_db/                       # FAISS vectorstores serialized per country
│   ├── spain_vectorstore.pkl
│   ├── france_vectorstore.pkl
│   ├── portugal_vectorstore.pkl
│   ├── germany_vectorstore.pkl
│   └── italy_vectorstore.pkl
├── test_translation_workflow.py     # Unit tests for translation and language detection components
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation (this file)
```

---

## Process Overview

### 1. **Document Preprocessing**
- Civil Codes must be placed in the `Civil Codes/` folder.
- `document_parser.py` loads each PDF using `PyPDF2`, extracts raw text, and cleans headers/footnotes.
- The text is chunked using the parameters defined in `config.py`. Chunking is character-based with optional overlaps to maintain context continuity between sections.

### 2. **Embedding and Vectorization**
- Each chunk is embedded using the `all-MiniLM-L6-v2` SentenceTransformer.
- FAISS indexes are created per country to isolate semantic retrieval and reduce irrelevant matches.
- The FAISS index is stored in a `vectorstore.pkl` file within `vector_db/`.
- To regenerate all vectorstores, run `main.py`.

### 3. **Query Pipeline (RAG)**
The query process is multi-stage:

1. **Language Detection**: 
   `translator.py` uses `langdetect` to identify the input language.

2. **Translation (if needed)**: 
   If the query is not in English, it is translated using `deep-translator`.

3. **Country Classification**:
   `country_classifier.py` attempts to classify which country's civil code is most likely relevant to the query using keyword mapping or text embedding similarity.

4. **Retrieval**:
   `rag_engine.py` performs a FAISS similarity search within the appropriate vectorstore and selects top-k matching chunks.

5. **Prompt Construction**:
   Retrieved context is injected into a prompt template and sent to Gemini (Google Generative AI) for final response generation.

6. **Result Translation (optional)**:
   The final answer can be translated back to the user’s original language if it differs from English.

### 4. **Web Interface**
- `streamlit_app.py` launches an interactive web interface.
- Users can:
  - Enter queries in any language.
  - View matched context excerpts.
  - Select a response language.
  - Trigger reclassification or retranslation manually if necessary.

### 5. **Testing**
- `test_translation_workflow.py` verifies the accuracy of language detection and translation logic.
- Extend tests to include classifier accuracy and FAISS retrieval quality using synthetic queries.

---

## Advanced Details

### Chunking Strategy
- Chunks are typically between 300–500 characters with a 50-character overlap.
- This ensures contextual continuity across paragraphs and reduces semantic loss during embedding.

### Prompt Template
- A dynamic prompt template is used, inserting both user question and retrieved legal text.
- Example:
  ```
  Context: {retrieved_passages}
  Question: {user_query}
  Answer in concise legal terms.
  ```

### FAISS Indexing
- FAISS uses L2 distance with cosine-normalized vectors.
- You can adjust the `k` parameter to return more results or switch to Approximate Nearest Neighbor for performance at scale.

### Translation Notes
- If the `langdetect` module returns a confidence below 0.8, fallback rules or default English handling is used.
- Translator fallback options are planned (e.g., using HuggingFace transformers locally if rate-limited).

---

## Installation

### 1. **Set up environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. **Environment Variables**
Create a `.env` file in the root directory with the following:

```env
GEMINI_API_KEY=your_google_generative_ai_key
```

---

## Usage

### Build Vectorstores
```bash
python main.py
```

### Launch Interface
```bash
streamlit run streamlit_app.py
```

---

## Supported Countries

- Spain
- France
- Portugal
- Germany
- Italy

Ensure each PDF follows the naming format: `Civil Code - {Country}.pdf`.

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## Notes

- If a Civil Code file is missing or named incorrectly, it will be skipped.
- Translation currently uses `deep-translator` and `langdetect`; these can be swapped for alternatives like Google Translate API or HuggingFace models.
- FAISS indexes are country-specific to avoid noise and improve performance.
- Gemini LLM is used via the `google-generativeai` package; you may adapt for OpenAI or Mistral if preferred.

---

## Example Queries

### Country-Specific Legal Interpretation
These queries target one specific country and require retrieval from the corresponding Civil Code.

- English:
"What are the conditions for terminating a lease agreement in Spain?"

- Spanish:
"¿Cuáles son los requisitos para disolver un contrato de arrendamiento en España?"

- German:
"Welche Pflichten hat ein Käufer laut deutschem Zivilgesetzbuch?"
(What obligations does a buyer have under the German Civil Code?)

- Portuguese:
"O que diz o Código Civil de Portugal sobre herança legítima?"
(What does the Portuguese Civil Code say about lawful inheritance?)

- French:
"Quelles sont les règles concernant la capacité juridique dans le Code civil français?"
(What are the rules regarding legal capacity in the French Civil Code?)

### Cross-Language Query with Translation
These showcase the system’s ability to detect and translate queries before processing:

- Italian query with English result:
"Quali sono i requisiti per la validità di un contratto secondo il codice civile italiano?"
(What are the validity requirements for a contract under the Italian Civil Code?)

- User writes in English about Germany:
"Does German law allow cancellation of a sales contract after delivery?"

### Jurisdiction Classification Challenge
These questions do not specify a country, triggering the classifier to route intelligently based on terms used.

- "What are the rules for adverse possession?"
→ Likely routed to France, Germany, or Italy depending on phrasing.

- "Who has guardianship rights in case of parental death?"
→ Could be routed to all, or prioritized to Spain or Portugal based on statistical match.

### Multilingual QA and Translation Return
Translate results back to user’s language after RAG response.

- User inputs:
"¿Qué dice el Código Civil sobre la nulidad matrimonial?"

= System retrieves Spanish content, answers in Spanish, and (optionally) provides English translation if selected in UI.

### Comparative Use Cases (Planned Feature)
Not yet implemented, but the system is capable of being extended to answer:

- "Compare how inheritance laws differ between France and Germany."

- "In which countries is a prenuptial agreement enforceable according to the civil code?"

