Mental Health Copilot
=====================

**An AI-powered Retrieval-Augmented Generation (RAG) application for mental health insights, crisis detection, and empathetic support using Reddit data.**

📌 Overview
-----------

Mental Health Copilot is a full-stack AI system designed to extract, process, and analyze real-world mental health discussions from Reddit.It combines **NLP preprocessing**, **FAISS vector search**, and **LLM-powered responses** with a focus on **crisis detection** and **safe user interaction**.

The goal is to demonstrate how large-scale unstructured mental health discussions can be transformed into **actionable insights** and **context-aware conversational support** while adhering to responsible AI practices.

✨ Features
----------

*   **Automated Reddit Data Pipeline**
    
    *   Scrapes posts and comments from multiple mental health–focused subreddits.
        
    *   Preprocesses text with tokenization, lemmatization, stopword removal, and sentiment analysis.
        
    *   Stores curated, structured data in **MongoDB**.
        
*   **Crisis Detection Module**
    
    *   Uses keyword and regex matching to flag urgent discussions.
        
    *   Dynamically modifies chatbot prompts for either **empathetic responses** or **urgent safety-first guidance**.
        
*   **RAG-powered Semantic Search & Chatbot**
    
    *   Generates embeddings with **Sentence-Transformers**.
        
    *   Uses **FAISS** for high-speed semantic retrieval.
        
    *   Integrates with **OpenAI GPT-3.5** for context-aware and emotionally sensitive replies.
        
*   **Backend API & Modular Orchestration**
    
    *   Built with **FastAPI** to serve the RAG pipeline.
        
    *   Custom reusable components in **Langflow** for visual workflow orchestration.
        
*   **Interactive & Safe Frontend**
    
    *   **Streamlit** chat interface with real-time responses.
        
    *   Displays crisis helpline resources prominently.
        
    *   Maintains session state and displays metadata for transparency.
        
*   **Exploratory Data Analysis (EDA)**
    
    *   Jupyter Notebook visualizations using **Matplotlib** and **Pandas**.
        
    *   Insights into sentiment trends, topic distribution, and user activity patterns.
        

🛠️ Tech Stack
--------------

**Backend & Machine Learning**

*   Python, FastAPI, OpenAI GPT-3.5
    
*   Sentence-Transformers, FAISS, NLTK, TextBlob
    

**Frontend & Prototyping**

*   Streamlit, Langflow
    

**Data & Storage**

*   MongoDB, Pandas, NumPy, Matplotlib
    

**Tools & Platforms**

*   Git, Docker, Jupyter Notebook
    

📂 Project Structure
--------------------

`Mental-Health-Copilot/  │  ├── 01-scrape-store.py        # Reddit data scraping and MongoDB storage  ├── 02-nlp-process-store.py   # NLP preprocessing, embeddings, FAISS indexing  ├── 04-backend-api.py         # FastAPI backend serving RAG responses  ├── 06-streamlit-app.py       # Frontend chat interface  ├── eda/                      # Jupyter Notebooks for exploratory data analysis  ├── requirements.txt          # Python dependencies  └── README.md                 # Project documentation   `

🚀 Getting Started
------------------

### 1️⃣ Clone the repository

`   git clone https://github.com/ankur-mali/Mental-Health-Copilot-RAG.git  cd Mental-Health-Copilot-RAG   `

### 2️⃣ Install dependencies

`   pip install -r requirements.txt   `

### 3️⃣ Configure environment variables

Create a .env file:

`   MONGODB_URI=your_mongodb_connection_string  OPENAI_API_KEY=your_openai_api_key   `

### 4️⃣ Run the pipeline

**Step 1: Scrape Reddit data**
`   python 01-scrape-store.py   `

**Step 2: Process data and create FAISS index**

`   python 02-nlp-process-store.py   `

**Step 3: Launch FastAPI backend**

`   uvicorn 04-backend-api:app --reload   `

**Step 4: Start Streamlit frontend**

` streamlit run 06-streamlit-app.py   `

📊 Example Dashboard
--------------------

Visualizations from EDA show sentiment trends, topic clusters, and user activity patterns, informing chatbot design and crisis detection logic.

⚠️ Disclaimer
-------------

This project is for **research and demonstration purposes only**. It is **not** a replacement for professional mental health advice.If you or someone you know is in crisis, please seek help from a qualified mental health professional or contact a crisis helpline.
