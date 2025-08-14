import os
import pymongo
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    min_text_length: int = 20
    max_text_length: int = 2000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    overlap: int = 50
    vector_dimension: int = 384


class MentalHealthDataProcessor:
    def __init__(self):
        self.config = ProcessingConfig()
        self._setup_mongodb()
        self._setup_nlp_tools()
        self._setup_embedding_model()

        # Mental health specific keywords for enhanced processing
        self.mental_health_keywords = {
            'anxiety': ['anxious', 'panic', 'worry', 'nervous', 'stressed', 'overwhelmed'],
            'depression': ['depressed', 'sad', 'hopeless', 'empty', 'worthless', 'low mood'],
            'therapy': ['therapist', 'counseling', 'therapy', 'treatment', 'medication'],
            'coping': ['coping', 'mindfulness', 'meditation', 'exercise', 'support'],
            'relationships': ['family', 'friends', 'partner', 'social', 'isolation', 'lonely'],
            'work_life': ['work', 'job', 'career', 'stress', 'burnout', 'balance']
        }

    def _setup_mongodb(self):
        """Connect to MongoDB"""
        self.mongo_client = pymongo.MongoClient(os.getenv("MONGODB_CS"))
        self.db = self.mongo_client["reddit_mental_health"]
        self.posts_collection = self.db["posts"]
        self.comments_collection = self.db["comments"]
        self.processed_collection = self.db["processed_chunks"]
        self.embeddings_collection = self.db["embeddings"]
        logger.info("MongoDB connection established")

    def _setup_nlp_tools(self):
        """Initialize NLP tools"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('tagsets', quiet=True)
            logger.info("NLTK data download completed")
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("NLP tools initialized")
        except Exception as e:
            logger.error(f"Error setting up NLP tools: {e}")

    def _setup_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Embedding model loaded: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")

    def analyze_dataset_statistics(self):
        """Analyze and visualize dataset statistics"""
        logger.info("Analyzing dataset statistics...")

        # Basic statistics
        posts_count = self.posts_collection.count_documents({})
        comments_count = self.comments_collection.count_documents({})
        crisis_posts = self.posts_collection.count_documents({"is_crisis_flagged": True})
        crisis_comments = self.comments_collection.count_documents({"is_crisis_flagged": True})

        # Subreddit distribution
        pipeline = [
            {"$group": {"_id": "$subreddit", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        subreddit_dist = list(self.posts_collection.aggregate(pipeline))

        # Text length analysis
        posts = list(self.posts_collection.find({}, {"title": 1, "body": 1, "subreddit": 1}))
        text_lengths = [len((post.get('title', '') + ' ' + post.get('body', '')).strip())
                        for post in posts]

        # Create analysis report
        analysis = {
            "total_posts": posts_count,
            "total_comments": comments_count,
            "crisis_content": {"posts": crisis_posts, "comments": crisis_comments},
            "subreddit_distribution": subreddit_dist,
            "text_length_stats": {
                "mean": np.mean(text_lengths),
                "median": np.median(text_lengths),
                "min": np.min(text_lengths),
                "max": np.max(text_lengths),
                "std": np.std(text_lengths)
            },
            "analysis_date": datetime.utcnow()
        }

        # Save analysis
        with open('dataset_analysis.json', 'w') as f:
            json.dump(analysis, f, default=str, indent=2)

        logger.info("Dataset analysis completed and saved")
        return analysis

    def clean_text(self, text: str) -> str:
        """Advanced text cleaning for mental health content"""
        if not text:
            return ""

        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove user mentions
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold formatting
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic formatting

        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove very short words (< 2 chars) except important ones
        words = text.split()
        words = [word for word in words if len(word) > 2 or word.lower() in ['i', 'me', 'my', 'am', 'is', 'be', 'do']]

        return ' '.join(words).strip()

    def extract_mental_health_features(self, text: str) -> Dict:
        """Extract mental health specific features from text"""
        text_lower = text.lower()
        features = {}

        # Keyword matching
        for category, keywords in self.mental_health_keywords.items():
            features[f'{category}_mentions'] = sum(1 for keyword in keywords if keyword in text_lower)

        # Sentiment analysis
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0.0
            features['sentiment_subjectivity'] = 0.0

        # Text complexity metrics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        features['sentence_count'] = len(sentences)
        features['word_count'] = len(words)
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        features['question_count'] = text.count('?')
        features['exclamation_count'] = text.count('!')

        return features

    def create_text_chunks(self, text: str, title: str = "") -> List[Dict]:
        """Create overlapping chunks from text for better embedding"""
        if not text or len(text) < self.config.min_text_length:
            return []

        # Combine title and text
        full_text = f"{title}. {text}".strip()

        # Clean text
        cleaned_text = self.clean_text(full_text)

        if len(cleaned_text) < self.config.min_text_length:
            return []

        chunks = []
        words = cleaned_text.split()

        # If text is short enough, return as single chunk
        if len(' '.join(words)) <= self.config.chunk_size:
            chunk_data = {
                'text': cleaned_text,
                'word_count': len(words),
                'features': self.extract_mental_health_features(cleaned_text)
            }
            chunks.append(chunk_data)
            return chunks

        # Create overlapping chunks
        chunk_words = self.config.chunk_size // 6  # Approximate words per chunk
        overlap_words = self.config.overlap // 6

        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_words, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])

            if len(chunk_text) >= self.config.min_text_length:
                chunk_data = {
                    'text': chunk_text,
                    'word_count': end_idx - start_idx,
                    'chunk_index': len(chunks),
                    'features': self.extract_mental_health_features(chunk_text)
                }
                chunks.append(chunk_data)

            start_idx += chunk_words - overlap_words

            if end_idx >= len(words):
                break

        return chunks

    def process_posts_and_comments(self):
        """Process all posts and comments into chunks"""
        logger.info("Processing posts and comments...")

        # Clear previous processed data
        self.processed_collection.delete_many({})

        processed_count = 0

        # Process posts
        logger.info("Processing posts...")
        posts_cursor = self.posts_collection.find({})

        for post in posts_cursor:
            try:
                title = post.get('title', '')
                body = post.get('body', '')

                chunks = self.create_text_chunks(body, title)

                for chunk in chunks:
                    processed_doc = {
                        'source_type': 'post',
                        'source_id': post['_id'],
                        'reddit_id': post.get('reddit_id'),
                        'subreddit': post.get('subreddit'),
                        'text': chunk['text'],
                        'features': chunk['features'],
                        'metadata': {
                            'score': post.get('score', 0),
                            'created_datetime': post.get('created_datetime'),
                            'is_crisis_flagged': post.get('is_crisis_flagged', False),
                            'chunk_index': chunk.get('chunk_index', 0),
                            'word_count': chunk['word_count']
                        },
                        'processed_at': datetime.utcnow()
                    }

                    self.processed_collection.insert_one(processed_doc)
                    processed_count += 1

            except Exception as e:
                logger.error(f"Error processing post {post.get('_id')}: {e}")

        # Process comments
        logger.info("Processing comments...")
        comments_cursor = self.comments_collection.find({})

        for comment in comments_cursor:
            try:
                body = comment.get('body', '')

                chunks = self.create_text_chunks(body)

                for chunk in chunks:
                    processed_doc = {
                        'source_type': 'comment',
                        'source_id': comment['_id'],
                        'reddit_id': comment.get('reddit_id'),
                        'post_id': comment.get('post_id'),
                        'text': chunk['text'],
                        'features': chunk['features'],
                        'metadata': {
                            'score': comment.get('score', 0),
                            'created_datetime': comment.get('created_datetime'),
                            'is_crisis_flagged': comment.get('is_crisis_flagged', False),
                            'chunk_index': chunk.get('chunk_index', 0),
                            'word_count': chunk['word_count'],
                            'depth': comment.get('depth', 0)
                        },
                        'processed_at': datetime.utcnow()
                    }

                    self.processed_collection.insert_one(processed_doc)
                    processed_count += 1

            except Exception as e:
                logger.error(f"Error processing comment {comment.get('_id')}: {e}")

        logger.info(f"Processing completed. Total chunks created: {processed_count}")
        return processed_count

    def generate_embeddings(self, batch_size: int = 100):
        """Generate embeddings for all processed chunks"""
        logger.info("Generating embeddings...")

        # Clear previous embeddings
        self.embeddings_collection.delete_many({})

        # Get all processed chunks
        total_chunks = self.processed_collection.count_documents({})
        logger.info(f"Generating embeddings for {total_chunks} chunks")

        processed_count = 0

        # Process in batches
        cursor = self.processed_collection.find({})

        batch_texts = []
        batch_docs = []

        for doc in cursor:
            batch_texts.append(doc['text'])
            batch_docs.append(doc)

            if len(batch_texts) == batch_size:
                # Generate embeddings for batch
                embeddings = self.embedding_model.encode(batch_texts)

                # Store embeddings
                for i, embedding in enumerate(embeddings):
                    embedding_doc = {
                        'source_doc_id': batch_docs[i]['_id'],
                        'source_type': batch_docs[i]['source_type'],
                        'subreddit': batch_docs[i].get('subreddit'),
                        'text': batch_docs[i]['text'],
                        'embedding': embedding.tolist(),
                        'features': batch_docs[i]['features'],
                        'metadata': batch_docs[i]['metadata'],
                        'generated_at': datetime.utcnow()
                    }

                    self.embeddings_collection.insert_one(embedding_doc)
                    processed_count += 1

                # Clear batch
                batch_texts = []
                batch_docs = []

                if processed_count % 1000 == 0:
                    logger.info(f"Generated embeddings for {processed_count}/{total_chunks} chunks")

        # Process remaining batch
        if batch_texts:
            embeddings = self.embedding_model.encode(batch_texts)

            for i, embedding in enumerate(embeddings):
                embedding_doc = {
                    'source_doc_id': batch_docs[i]['_id'],
                    'source_type': batch_docs[i]['source_type'],
                    'subreddit': batch_docs[i].get('subreddit'),
                    'text': batch_docs[i]['text'],
                    'embedding': embedding.tolist(),
                    'features': batch_docs[i]['features'],
                    'metadata': batch_docs[i]['metadata'],
                    'generated_at': datetime.utcnow()
                }

                self.embeddings_collection.insert_one(embedding_doc)
                processed_count += 1

        logger.info(f"Embedding generation completed. Total embeddings: {processed_count}")
        return processed_count

    def create_faiss_index(self):
        """Create FAISS index for fast similarity search"""
        logger.info("Creating FAISS index...")

        # Get all embeddings
        embeddings_cursor = self.embeddings_collection.find({})

        embeddings = []
        doc_ids = []

        for doc in embeddings_cursor:
            embeddings.append(doc['embedding'])
            doc_ids.append(str(doc['_id']))

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add embeddings to index
        index.add(embeddings_array)

        # Save index and mapping
        faiss.write_index(index, "mental_health_embeddings.faiss")

        with open("embedding_doc_mapping.pkl", "wb") as f:
            pickle.dump(doc_ids, f)

        logger.info(f"FAISS index created with {len(embeddings)} vectors")
        return index, doc_ids

    def create_visualization_dashboard(self):
        """Create visualizations for dataset analysis"""
        logger.info("Creating visualization dashboard...")

        # Subreddit distribution
        pipeline = [
            {"$group": {"_id": "$subreddit", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        subreddit_data = list(self.processed_collection.aggregate(pipeline))

        plt.figure(figsize=(15, 10))

        # Subreddit distribution
        plt.subplot(2, 3, 1)
        subreddits = [item['_id'] for item in subreddit_data]
        counts = [item['count'] for item in subreddit_data]
        plt.bar(subreddits, counts)
        plt.title('Content Distribution by Subreddit')
        plt.xticks(rotation=45)

        # Sentiment distribution
        plt.subplot(2, 3, 2)
        sentiments = []
        processed_docs = self.processed_collection.find({})
        for doc in processed_docs:
            sentiment = doc.get('features', {}).get('sentiment_polarity', 0)
            sentiments.append(sentiment)

        plt.hist(sentiments, bins=30, alpha=0.7)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Polarity (-1 to 1)')

        # Word count distribution
        plt.subplot(2, 3, 3)
        word_counts = [doc.get('features', {}).get('word_count', 0)
                       for doc in self.processed_collection.find({})]
        plt.hist(word_counts, bins=30, alpha=0.7)
        plt.title('Word Count Distribution')
        plt.xlabel('Words per Chunk')

        # Crisis content percentage
        plt.subplot(2, 3, 4)
        crisis_count = self.processed_collection.count_documents({'metadata.is_crisis_flagged': True})
        total_count = self.processed_collection.count_documents({})
        crisis_percentage = (crisis_count / total_count) * 100

        labels = ['Regular Content', 'Crisis-Flagged Content']
        sizes = [100 - crisis_percentage, crisis_percentage]
        colors = ['lightblue', 'lightcoral']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Crisis Content Detection')

        # Mental health keyword frequency
        plt.subplot(2, 3, 5)
        keyword_counts = {}
        for category in self.mental_health_keywords.keys():
            count = self.processed_collection.count_documents({
                f'features.{category}_mentions': {'$gt': 0}
            })
            keyword_counts[category] = count

        categories = list(keyword_counts.keys())
        counts = list(keyword_counts.values())
        plt.bar(categories, counts)
        plt.title('Mental Health Topic Frequency')
        plt.xticks(rotation=45)

        # Time-based posting patterns (if date available)
        plt.subplot(2, 3, 6)
        # Simple hour-based pattern (mock data for now)
        hours = list(range(24))
        activity = np.random.poisson(50, 24)  # Replace with real data analysis
        plt.plot(hours, activity)
        plt.title('Daily Posting Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Post Count')

        plt.tight_layout()
        plt.savefig('mental_health_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Visualization dashboard created and saved")

    def run_complete_pipeline(self):
        """Run the complete data processing pipeline"""
        logger.info("Starting complete data processing pipeline...")

        # Step 1: Analyze dataset
        analysis = self.analyze_dataset_statistics()

        # Step 2: Process data into chunks
        chunk_count = self.process_posts_and_comments()

        # Step 3: Generate embeddings
        embedding_count = self.generate_embeddings()

        # Step 4: Create FAISS index
        index, doc_ids = self.create_faiss_index()

        # Step 5: Create visualizations
        self.create_visualization_dashboard()

        # Summary
        summary = {
            "pipeline_completion": datetime.utcnow(),
            "chunks_created": chunk_count,
            "embeddings_generated": embedding_count,
            "faiss_index_size": len(doc_ids),
            "ready_for_rag": True
        }

        with open('processing_summary.json', 'w') as f:
            json.dump(summary, f, default=str, indent=2)

        logger.info("Complete pipeline finished successfully!")
        logger.info(f"Created {chunk_count} chunks with {embedding_count} embeddings")
        logger.info("Ready for RAG implementation!")

        return summary

    def close_connections(self):
        """Clean up connections"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            logger.info("Database connections closed")


def main():
    processor = MentalHealthDataProcessor()

    try:
        summary = processor.run_complete_pipeline()
        print("\n" + "=" * 60)
        print("DATA PROCESSING PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Chunks Created: {summary['chunks_created']}")
        print(f"Embeddings Generated: {summary['embeddings_generated']}")
        print(f"FAISS Index Size: {summary['faiss_index_size']}")
        print(f"Status: {'✅ Ready for RAG Implementation' if summary['ready_for_rag'] else '❌ Issues Found'}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        processor.close_connections()


if __name__ == "__main__":
    main()