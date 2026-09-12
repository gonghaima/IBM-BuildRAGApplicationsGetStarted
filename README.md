# Summarize Private Documents Using RAG, LangChain, and LLMs

This project is a hands-on lab from IBM's "Build RAG Applications" course
(Skills Network). It walks through building a document question-answering
and summarization application using Retrieval-Augmented Generation (RAG),
[LangChain](https://www.langchain.com/), and an LLM served through
[IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai).

## What it does

The notebook builds a small RAG pipeline end-to-end:

1. **Load** a plain-text document (`companyPolicies.txt`, a set of sample
   company policies) with a LangChain `TextLoader`.
2. **Split** the document into chunks with a `CharacterTextSplitter`.
3. **Embed** the chunks using Hugging Face sentence embeddings and store
   them in a [Chroma](https://www.trychroma.com/) vector store.
4. **Retrieve & generate** answers with an LLM
   (`meta-llama/llama-4-maverick-17b-128e-instruct-fp8` via watsonx.ai)
   using LangChain's `RetrievalQA` chain.
5. Improve the app with a **prompt template** that stops the model from
   hallucinating answers not found in the source document.
6. Add **conversation memory** with `ConversationBufferMemory` and
   `ConversationalRetrievalChain` so follow-up questions ("What I cannot
   do in it?") resolve correctly against prior chat history.
7. Wrap everything into a simple interactive **chat agent** you can query
   from the notebook (type `quit`, `exit`, or `bye` to stop it).

The notebook also includes three optional exercises: using a different
source document, returning the source chunks alongside answers, and
swapping in a different watsonx.ai model (e.g. Mistral).

## Files

- `SummarizePrivateDocumentsusingRAGLangChainandLLMs.ipynb` — the lab notebook.
- `companyPolicies.txt` — sample document (fictional company policies) used
  as the RAG data source.

## Running it

This notebook is designed to run in IBM's Skills Network Labs environment,
which provides preconfigured watsonx.ai credentials. To run it elsewhere,
you'll need:

- Python with Jupyter
- The packages installed in the first code cell: `langchain`,
  `langchain-community`, `langchain-chroma`, `langchain-huggingface`,
  `langchain-classic`, `langchain-ibm`, `ibm-watsonx-ai`, `chromadb`,
  `sentence-transformers`, `transformers`, `huggingface-hub`, `wget`
- Your own IBM watsonx.ai `credentials` and `project_id` (the notebook uses
  placeholder values that only work inside the lab environment)
