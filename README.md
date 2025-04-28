# 🌐 WebRAG Agent

**WebRAG Agent** is a Retrieval-Augmented Generation (RAG) application that fetches information from a website, processes it into retrievable chunks, and answers user questions based strictly on the web content.

---

## 🚀 Features

- 🌍 **Web Scraping:**  
  Automatically loads and parses content from a live webpage.

- 📖 **Document Splitting:**  
  Breaks large text into smaller, overlapping chunks for better retrieval performance.

- 🔎 **Vector Search (ChromaDB):**  
  Creates vector embeddings for each chunk using HuggingFace models.

- 🤖 **Contextual Answer Generation:**  
  Uses OpenAI's GPT-4o-mini model to answer questions based on the retrieved context.

- 🧩 **RAG Prompting:**  
  Utilizes a retrieval-augmented generation (RAG) prompt template to ensure responses stay grounded in the provided content.

- 🌀 **Streaming Responses:**  
  Outputs the answer piece by piece as it's being generated.

---

## 📦 Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma Vector Database](https://docs.trychroma.com/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [OpenAI GPT-4o-mini](https://platform.openai.com/)
- [BeautifulSoup (bs4)](https://www.crummy.com/software/BeautifulSoup/)
- [LangChain Hub Prompts](https://smith.langchain.com/hub)

---

⚙️ How It Works
Load Web Content:
It scrapes specific sections (post-content, post-title, etc.) from the target URL.

Text Processing:

The scraped content is split into smaller overlapping chunks.

Each chunk is embedded into a vector space using HuggingFace's all-MiniLM-L6-v2 model.

RAG Pipeline:

When a user asks a question, the system retrieves the most relevant chunks.

It then prompts the LLM with the retrieved context and the question.

The model generates a context-aware answer.

Streaming Output:

The answer is printed in real time as it is being generated.
