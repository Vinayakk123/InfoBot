# INFOBOT — Project Summary

## 1. What this is

INFOBOT is a chatbot that answers questions about InfoBeans Technologies using only the company's own documents — things like PDFs, spreadsheets, and text files. Instead of relying on whatever a general AI model happens to "know," it reads the actual source documents first and builds its answers strictly from them. Think of it as a smart search assistant that has read a specific stack of company paperwork and will only answer using what's actually written in that paperwork.

## 2. The problem it solves

Normally, if someone wants to know something buried in a company document — a policy, a fact from a report, a detail in a spreadsheet — they have to manually search through files, open several documents, and piece the answer together themselves. This is slow, easy to get wrong, and doesn't scale as the number of documents grows. INFOBOT replaces that manual digging with a simple chat: ask a question in plain English, get an answer pulled directly from the relevant documents, along with a reference to exactly where that answer came from.

## 3. How it works

- Documents (PDFs, Word files, spreadsheets, text files, etc.) are dropped into a folder and the system reads all of them automatically.
- Each document is broken into small, manageable chunks of text, since it's easier to search and match small pieces than whole documents.
- Every chunk is converted into a kind of numerical "fingerprint" that captures its meaning, not just its exact wording — this is done locally on the machine, so document content isn't sent anywhere just to prepare it for search.
- Those fingerprints are stored in a searchable index, essentially a fast lookup system built for finding "meaning-similar" text rather than exact keyword matches.
- When someone asks a question, the system finds the most relevant chunks of text from that index, hands them to an AI language model, and asks it to answer using only that material — if the documents don't contain the answer, it says so rather than guessing.
- The chat interface shows the answer along with its sources (which file and page/row it came from), so the user can verify where the information came from.

## 4. What it's built with

- A web-based chat interface (built with Streamlit, a tool for building simple interactive apps)
- A document-understanding toolkit (LangChain) that handles reading different file types
- A local, on-device model for turning text into searchable "fingerprints" (no external service needed for this step)
- A fast search index (FAISS, a library built for this kind of similarity search) to store and retrieve those fingerprints
- A cloud-hosted AI language model (via Groq) that generates the actual answers
- Everything is written in Python

## 5. What makes it production-ready

- Answers are always grounded in cited sources — every response shows which document and section it came from, so answers can be checked, not just trusted blindly.
- The system won't fabricate an answer when nothing relevant is found — it explicitly says so instead of guessing.
- Document loading is resilient: if one file fails to load, the rest still process normally instead of the whole system breaking.
- The chat app has user-facing error handling, so failures show a readable message instead of crashing.
- Conversation history is saved locally so a session can be picked back up rather than lost on restart.
- Secrets (API keys) are kept out of the codebase and excluded from version control.

## 6. Current limitations

- This is a working prototype, not a hardened production system: there are no automated tests, no packaging for easy deployment (like Docker), and no automated build/check pipeline yet.
- There's no login or access control — anyone who can reach the chat app can use it and consume the shared AI usage quota.
- It's built for one person using it at a time on one machine, not many simultaneous users.
- The AI model and the "fingerprinting" model are currently fixed choices, set in the code rather than easily swappable.
- Its search approach compares every stored chunk directly, which works well at today's small document scale but would need upgrading for a much larger document library.

## 7. What's next

- Harden it for real deployment: add automated tests, package it (e.g., with Docker), and set up basic access control.
- Make the AI model and search settings configurable, so the system isn't locked into one specific model or provider.
- Scale the document search approach to comfortably handle a much larger and continually growing set of documents.
