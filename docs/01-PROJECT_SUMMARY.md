# INFOBOT — Project Summary

*Business-facing overview. For technical depth on any item below, see [03-TECH_STACK.md](03-TECH_STACK.md).*

## 1. What this is

INFOBOT is a chatbot that answers questions about InfoBeans Technologies using only the company's own documents — PDFs, spreadsheets, Word files, and text files. Instead of relying on whatever a general AI model happens to "know," it reads the actual source documents first and builds its answers strictly from them. Think of it as a search assistant that has read a specific stack of company paperwork and will only answer using what's actually written in it.

## 2. The problem it solves

Normally, finding something buried in a company document — a policy, a fact from a report, a detail in a spreadsheet — means manually opening files and piecing the answer together. That's slow, error-prone, and doesn't scale as documents pile up. INFOBOT replaces that manual digging with a simple chat: ask a question in plain English, get an answer pulled directly from the relevant documents, plus a reference to exactly where that answer came from.

## 3. How it works

- Documents (PDF, Word, Excel, CSV, TXT, JSON) are dropped into a folder and the system reads all of them automatically.
- Each document is broken into small, overlapping chunks of text — smaller pieces are easier to search and match than whole documents.
- Every chunk is converted into a numerical "fingerprint" that captures its meaning, not just its exact wording. This happens locally on the machine — no document content is sent to an external service for this step.
- Those fingerprints are stored in a searchable index built for finding "meaning-similar" text rather than exact keyword matches.
- When someone asks a question, the system finds the most relevant chunks, hands them to an AI language model, and asks it to answer using only that material. If the documents don't contain the answer, it says so rather than guessing.
- The chat interface shows the answer alongside its sources (which file and page/row it came from), so answers can be verified, not just trusted.

## 4. What it's built with

- A web-based chat interface (Streamlit)
- A document-understanding toolkit (LangChain) that handles reading different file types
- A local, on-device model that turns text into searchable "fingerprints" (no external service needed for this step)
- A fast search index (FAISS) to store and retrieve those fingerprints
- A cloud-hosted AI language model (via Groq) that generates the actual answers
- Everything runs in Python, packaged to run in a single Docker container

Full technical detail, including exact model names and swap-out alternatives, is in [03-TECH_STACK.md](03-TECH_STACK.md).

## 5. What makes it production-ready

- Answers are always grounded in cited sources — every response shows which document and section it came from.
- The system won't fabricate an answer when nothing relevant is found — it says so explicitly instead of guessing.
- Document loading is resilient: if one file fails to load, the rest still process normally instead of the whole system breaking.
- The chat app has user-facing error handling, so failures show a readable message instead of crashing.
- Conversation history is saved locally so a session can be picked back up rather than lost on restart.
- Secrets (API keys) are kept out of the codebase and out of version control.
- The whole app is packaged as a Docker image with a health check, so it can be deployed the same way anywhere Docker runs.

## 6. Current limitations

- This is a working demo, not a hardened multi-user production system: no automated tests, no login/access control, and no automated build/check pipeline yet.
- It's designed for one person using it at a time on one machine, not many simultaneous users.
- The AI model and the "fingerprinting" model are currently fixed choices, set in the code rather than exposed as configuration.
- The search approach compares every stored chunk directly (no approximate search), which works well at today's small document scale but would need upgrading for a much larger document library.
- Everything (chat UI, retrieval, and the call to the AI model) runs in a single container/process — a reasonable choice at this stage, but a natural next step is splitting it into a separate backend API and frontend. See [02-ARCHITECTURE.md](02-ARCHITECTURE.md) for detail.

## 7. What's next

- Add automated tests and basic access control (login) before any shared/public deployment.
- Make the AI model and search settings configurable, so the system isn't locked into one specific model or provider.
- Scale the document search approach to comfortably handle a much larger and continually growing set of documents.
- Split the single container into a separate API service and frontend as usage grows.
