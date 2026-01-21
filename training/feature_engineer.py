import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import faiss
from typing import Tuple

class FeatureEngineer:
    """Handles vectorization and dimensionality reduction"""
    
    def __init__(self, n_components: int = 256, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.tfidf = None
        self.svd = None
    
    def create_tfidf_vectors(self, texts: pd.Series) -> np.ndarray:
        """
        Create TF-IDF vectors from text
        
        Args:
            texts: Series of text documents
            
        Returns:
            TF-IDF matrix (sparse)
        """
        self.tfidf = TfidfVectorizer(analyzer='word', stop_words='english', max_features=5000)
        return self.tfidf.fit_transform(texts)
    
    def reduce_dimensions(self, X_tfidf) -> np.ndarray:
        """
        Reduce dimensions using SVD
        
        Args:
            X_tfidf: TF-IDF matrix
            
        Returns:
            Reduced embeddings (float32)
        """
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        X_reduced = self.svd.fit_transform(X_tfidf)
        
        return X_reduced.astype('float32')
    
    def create_embeddings(self, texts: pd.Series) -> np.ndarray:
        """
        Create embeddings from text (TF-IDF + SVD)
        
        Args:
            texts: Series of text documents
            
        Returns:
            Embeddings (normalized)
        """
        import faiss
        # Create TF-IDF vectors
        X_tfidf = self.create_tfidf_vectors(texts)
        # Reduce dimensions
        embeddings = self.reduce_dimensions(X_tfidf)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def get_tfidf_vectorizer(self):
        """Get the fitted TF-IDF vectorizer"""
        return self.tfidf
    
    def get_svd_model(self):
        """Get the fitted SVD model"""
        return self.svd