import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import torch
import os
import nltk
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Singleton pattern for model loading to avoid loading it multiple times
class MLSummarizer:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the ML summarizer with a sentence transformer model"""
        self.initialized = False
        self.model = None
        
        # Try to initialize the model
        try:
            # Use a smaller, faster model that works well on Intel Macs
            # MiniLM models are much smaller and faster than larger models
            model_name = 'all-MiniLM-L6-v2'  # ~80MB, very fast, good performance
            self.model = SentenceTransformer(model_name)
            self.initialized = True
            print(f"Successfully loaded {model_name} model")
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}")
            # If loading fails, we'll fall back to non-ML methods

    def is_initialized(self) -> bool:
        """Check if the model was initialized successfully"""
        return self.initialized and self.model is not None

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences into embeddings"""
        if not self.is_initialized():
            raise ValueError("Model not initialized")
        
        # Convert sentences to embeddings using the transformer model
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        return embeddings
    
    def extract_summary_kmeans(self, text: str, num_sentences: int = 5) -> str:
        """Extract summary using K-means clustering of sentence embeddings"""
        if not text or len(text) < 100:
            return text
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Encode sentences
        embeddings = self.encode_sentences(sentences)
        
        # Use K-means clustering to find the most representative sentences
        num_clusters = min(num_sentences, len(sentences))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        
        # Find sentences closest to the centroids
        closest_indices = []
        for i in range(num_clusters):
            distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[i], axis=1)
            closest_idx = np.argmin(distances)
            closest_indices.append(closest_idx)
        
        # Sort by order of appearance in the original text
        closest_indices.sort()
        
        # Return the selected sentences in their original order
        selected_sentences = [sentences[idx] for idx in closest_indices]
        return ' '.join(selected_sentences)

    def summarize_long_text(self, text: str, max_length: int = 1000) -> str:
        """Summarize long text by chunking it and applying the ML summarization"""
        if not text:
            return "No text available for summarization."
        
        try:
            # If model is not initialized, return a truncated version
            if not self.is_initialized():
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # For short texts, summarize directly
            if len(text) < 5000:
                return self.extract_summary_kmeans(text, num_sentences=max(3, max_length//150))
            
            # For longer texts, chunk and process
            chunk_size = 4000  # Smaller chunks for better processing
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                if len(chunk) > 100:  # Only process meaningful chunks
                    summary = self.extract_summary_kmeans(chunk, num_sentences=3)
                    chunk_summaries.append(summary)
            
            # Combine summaries
            combined_text = ' '.join(chunk_summaries)
            
            # If still too long, summarize again
            if len(combined_text) > max_length:
                final_summary = self.extract_summary_kmeans(
                    combined_text, num_sentences=max(3, max_length//150)
                )
                return final_summary
            else:
                return combined_text
            
        except Exception as e:
            print(f"Error in ML summarization: {e}")
            # Fallback to basic truncation
            return text[:max_length] + "..." if len(text) > max_length else text

# Utility functions for the rest of the application to use
def get_ml_summarizer() -> MLSummarizer:
    """Get the singleton ML summarizer instance"""
    return MLSummarizer.get_instance()

def summarize_text_ml(text: str, max_length: int = 1000) -> str:
    """Summarize text using the ML-based summarizer"""
    summarizer = get_ml_summarizer()
    return summarizer.summarize_long_text(text, max_length)

def generate_content_summary(sources: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
    """Generate a more detailed content summary using ML techniques"""
    summarizer = get_ml_summarizer()
    
    # Extract and combine text from sources
    combined_text = ""
    for source in sources:
        if 'summary' in source:
            combined_text += source['summary'] + " "
        elif 'text' in source:
            combined_text += source['text'] + " "
    
    if not combined_text:
        return {
            "summary": f"No content available for {topic}",
            "key_points": [f"No data available for {topic}"],
            "topic": topic
        }
    
    # Generate a summary
    summary = summarizer.summarize_long_text(combined_text, max_length=1500)
    
    # Extract key points (individual sentences from the summary)
    sentences = sent_tokenize(summary)
    key_points = [f"- {sentence}" for sentence in sentences[:5]]
    
    # Add topic-specific points if we have very few
    if len(key_points) < 3:
        key_points.extend([
            f"- {topic} represents an important area of study",
            f"- Further research on {topic} is recommended",
            f"- Applications of {topic} span multiple domains"
        ])
    
    return {
        "summary": summary,
        "key_points": key_points,
        "topic": topic
    } 