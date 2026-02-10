#  AI Clinic Patient Assistant (RAG)

A Retrieval-Augmented Generation (RAG) system designed to serve as a virtual assistant for **HealthFirst Clinic**. This bot uses semantic search to answer patient questions accurately by referencing official clinic policy documents.

##  Key Features
- **PDF Intelligence**: Automatically extracts and chunks text from clinic policy PDFs.
- **Semantic Search**: Uses `sentence-transformers` and `FAISS` to find relevant policy sections based on the meaning of a user's question.
- **Context-Aware Answers**: Powered by OpenRouter APIs to provide conversational responses grounded only in provided clinic data.

## Tech Stack
- **Language**: Python 3.14
- **AI/ML**: `sentence-transformers` (all-MiniLM-L6-v2), `FAISS` (Vector Database)
- **APIs**: OpenRouter (LLM access)
- **Data Processing**: `pypdf`, `numpy`

##  Clinic Policy Logic (RAG Context)
The assistant is trained on specific rules from the `clinic_policies.pdf`:
- **Late Arrivals**: Arrive **15 minutes early**; arrivals over **15 minutes late** may be rescheduled.
- **Fees**: **$50 fee** for late cancellations (<24hr); **full visit cost** for no-shows.
- **Hours**: Open Mon-Fri (8 AM - 6 PM) and Sat (9 AM - 1 PM).

##  Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/NadeenMahmoud-DSAI/chatbot_OpenRouter_API.git](https://github.com/NadeenMahmoud-DSAI/chatbot_OpenRouter_API.git)
