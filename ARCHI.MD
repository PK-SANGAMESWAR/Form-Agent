## Architecture

------------------------------------------------------------------
Pipelines Explored

During development, three different pipelines were evaluated for the Intelligent Form Agent. The final architecture was chosen based on simplicity, reliability, offline capability, and deterministic performance for structured forms.

The Intelligent Form Agent is designed to ingest forms (PDF and TXT), extract structured metadata and unstructured text, index documents for retrieval, and answer natural language questions.

A key design goal is 100% local execution with no external API dependencies. Every component includes fallback mechanisms so the system continues to function even when deep learning models are unavailable.


                    ┌──────────────┐
Form Files ───────► │  Ingestion   │
                    │ ingestion.py │
                    │ (PDF / TXT)  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Extraction   │
                    │ extraction.py│
                    │ Regex →      │
                    │ Structured + │
                    │ Free Text    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ FormAgent    │
                    │  agent.py    │
                    │ Orchestrator │
                    └───┬────┬─────┘
                        │    │
          ┌─────────────┘    └─────────────┐
          ▼                                ▼
 ┌────────────────┐               ┌────────────────┐
 │   models.py    │               │ indexing.py    │
 │ QA             │               │ Semantic Search│
 │ Summarization  │               │ Embeddings /   │
 │ TF-IDF Fallback│               │ TF-IDF Fallback│
 └────────────────┘               └────────────────┘
                 │
                 ▼
         ┌────────────────┐
         │ app.py / CLI   │
         │ Streamlit UI   │
         └────────────────┘


## using distilbert-base-cased-distilled-squad - Extractive QA direcly from the data

The main arch is  - 
PDD/TXT -> Regex Rules -> Structured field, Free Text -> Embed Model -> Semantic Search -> DistilBertQA 

## PROBLEM - Indirect Question Answering capability is not there 

---------------------------------------------------------

Document
      │
      ▼
Ingestion
      │
      ▼
Extraction
      │
      ├──────────────┐
      ▼              ▼
Structured Store   Vector Store
(SQLite)           (ChromaDB)
      │              │
      └──────┬───────┘
             ▼
          Router
             ▼
        Retrieval
             ▼
        LLM Synthesis
             ▼
          Final Answer

