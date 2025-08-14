import numpy as np
import faiss
import pickle
import pymongo
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import json
import re
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    top_k_results: int = 5
    max_context_length: int = 2000
    temperature: float = 0.3
    crisis_threshold: float = 0.7


class MentalHealthRAGSystem:
    def __init__(self):
        self.config = RAGConfig()
        self._setup_models()
        self._setup_database()
        self._load_faiss_index()

        # Crisis detection keywords
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'self harm', 'cutting',
            'overdose', 'no point', 'better off dead', 'want to die', 'ending it'
        ]

        # System prompts for different scenarios
        self.system_prompts = {
            'general': """You are a compassionate mental health support assistant. Your responses should be:
1. Empathetic and understanding
2. Based on real community experiences from Reddit
3. Non-judgmental and supportive
4. Include disclaimers that you're not a replacement for professional help
5. Encourage seeking professional support when appropriate

Use the provided context from Reddit discussions to give evidence-based, community-driven insights.""",

            'crisis': """CRISIS DETECTED: You are responding to someone who may be in mental health crisis.
Your response should:
1. Express immediate concern and empathy
2. Strongly encourage contacting crisis resources immediately
3. Provide crisis hotline numbers: National Suicide Prevention Lifeline: 988
4. Avoid giving detailed advice - focus on immediate safety
5. Be warm but direct about seeking immediate professional help

Use the context to show they're not alone, but prioritize immediate safety."""
        }

    def _setup_models(self):
        """Initialize embedding and LLM models"""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Embedding model loaded: {self.config.embedding_model}")

            # Setup OpenAI client (you can replace with local models)
            self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("LLM client initialized")

        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise

    def _setup_database(self):
        """Setup MongoDB connection"""
        self.mongo_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.mongo_client["reddit_mental_health"]
        self.embeddings_collection = self.db["embeddings"]
        self.conversations_collection = self.db["conversations"]
        logger.info("Database connection established")

    def _load_faiss_index(self):
        """Load FAISS index and document mapping"""
        try:
            self.faiss_index = faiss.read_index("mental_health_embeddings.faiss")

            with open("embedding_doc_mapping.pkl", "rb") as f:
                self.doc_id_mapping = pickle.load(f)

            logger.info(f"FAISS index loaded with {len(self.doc_id_mapping)} documents")

        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def detect_crisis_intent(self, query: str) -> Tuple[bool, float]:
        """Detect if query indicates mental health crisis"""
        query_lower = query.lower()

        # Keyword-based detection
        crisis_score = 0.0
        for keyword in self.crisis_keywords:
            if keyword in query_lower:
                crisis_score += 0.2

        # Additional pattern matching
        crisis_patterns = [
            r'\bi\s+want\s+to\s+die\b',
            r'\bkill\s+myself\b',
            r'\bend\s+it\s+all\b',
            r'\bno\s+point\s+in\s+living\b',
            r'\bbetter\s+off\s+dead\b'
        ]

        for pattern in crisis_patterns:
            if re.search(pattern, query_lower):
                crisis_score += 0.3

        is_crisis = crisis_score >= self.config.crisis_threshold

        return is_crisis, crisis_score

    def retrieve_relevant_context(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents using FAISS similarity search"""
        if top_k is None:
            top_k = self.config.top_k_results

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding, dtype=np.float32)

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, top_k)

            # Retrieve documents from MongoDB
            relevant_docs = []

            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= len(self.doc_id_mapping):
                    continue

                doc_id = self.doc_id_mapping[idx]

                # Get document from MongoDB
                doc = self.embeddings_collection.find_one({"_id": pymongo.ObjectId(doc_id)})

                if doc:
                    relevant_docs.append({
                        'text': doc['text'],
                        'similarity': float(similarity),
                        'subreddit': doc.get('subreddit', 'unknown'),
                        'source_type': doc.get('source_type', 'unknown'),
                        'features': doc.get('features', {}),
                        'metadata': doc.get('metadata', {})
                    })

            return relevant_docs

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []

    def create_context_string(self, relevant_docs: List[Dict]) -> str:
        """Create context string from retrieved documents"""
        if not relevant_docs:
            return "No relevant community discussions found."

        context_parts = []
        total_length = 0

        for i, doc in enumerate(relevant_docs):
            # Format document with metadata
            subreddit = doc.get('subreddit', 'unknown')
            source_type = doc.get('source_type', 'post')
            sentiment = doc.get('features', {}).get('sentiment_polarity', 0)

            sentiment_label = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"

            doc_text = f"[Community {source_type} from r/{subreddit} - {sentiment_label} sentiment]\n{doc['text']}\n"

            # Check if adding this document exceeds context limit
            if total_length + len(doc_text) > self.config.max_context_length:
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n---\n".join(context_parts)

    def generate_response(self, query: str, context: str, is_crisis: bool = False) -> str:
        """Generate response using LLM with retrieved context"""
        try:
            # Choose system prompt based on crisis detection
            system_prompt = self.system_prompts['crisis'] if is_crisis else self.system_prompts['general']

            # Create user prompt
            user_prompt = f"""Context from Reddit mental health communities:
{context}

User Question: {query}

Please provide a helpful, empathetic response based on the community experiences above. Always remind users that this is for informational purposes and not a substitute for professional mental health care."""

            # Generate response
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please consider reaching out to a mental health professional if you need immediate support."

    def process_query(self, query: str, user_id: Optional[str] = None) -> Dict:
        """Main query processing function"""
        try:
            # Detect crisis
            is_crisis, crisis_score = self.detect_crisis_intent(query)

            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)

            # Create context string
            context = self.create_context_string(relevant_docs)

            # Generate response
            response = self.generate_response(query, context, is_crisis)

            # Log conversation (optional)
            conversation_log = {
                'user_id': user_id,
                'query': query,
                'is_crisis': is_crisis,
                'crisis_score': crisis_score,
                'response': response,
                'relevant_docs_count': len(relevant_docs),
                'timestamp': datetime.utcnow()
            }

            # Store conversation (optional - for analytics)
            if user_id:
                self.conversations_collection.insert_one(conversation_log)

            return {
                'response': response,
                'is_crisis': is_crisis,
                'crisis_score': crisis_score,
                'sources_used': len(relevant_docs),
                'context_length': len(context)
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please reach out to a mental health professional if you need immediate support.",
                'is_crisis': False,
                'crisis_score': 0.0,
                'sources_used': 0,
                'context_length': 0
            }

    def close_connections(self):
        """Clean up database connections"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()


# FastAPI endpoint for Langflow integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mental Health RAG API")
rag_system = MentalHealthRAGSystem()


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    is_crisis: bool
    crisis_score: float
    sources_used: int


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process mental health query endpoint"""
    try:
        result = rag_system.process_query(request.query, request.user_id)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)