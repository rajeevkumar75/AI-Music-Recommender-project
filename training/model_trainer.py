import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd
from training.data_processor import DataProcessor
from training.feature_engineer import FeatureEngineer

class ModelTrainer:
    """Orchestrates the training pipeline"""
    
    def __init__(
        self,
        data_csv_path: str,
        output_dir: str = "models",
        sample_size: Optional[int] = None,
        n_components: int = 256
    ):
        self.data_csv_path = data_csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.sample_size = sample_size
        self.n_components = n_components
        
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer(n_components=n_components)
        
        self.df = None
        self.embeddings = None
        self.faiss_index = None
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean data"""
        print("Loading data...")
        self.df = self.data_processor.load_data(
            self.data_csv_path,
            sample_size=self.sample_size
        )
        
        print("Checking data quality...")
        self.data_processor.check_data_quality(self.df)
        
        print("Preprocessing text...")
        self.df = self.data_processor.preprocess_text_column(self.df, column='text')
        
        return self.df
    
    def create_embeddings(self) -> np.ndarray:
        """Create embeddings from data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_clean_data() first.")
        
        print("Creating TF-IDF vectors...")
        print("Reducing dimensions...")
        
        self.embeddings = self.feature_engineer.create_embeddings(self.df['text'])
        
        print(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def build_faiss_index(self) -> faiss.IndexFlatL2:
        """Build FAISS index from embeddings"""
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first.")
        
        print("Building FAISS index...")
        
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)
        
        print(f"FAISS index created with {self.faiss_index.ntotal} vectors")
        return self.faiss_index
    
    def save_models(self):
        """Save all models and data"""
        if self.df is None or self.embeddings is None or self.faiss_index is None:
            raise ValueError("Models not fully trained. Complete training first.")
        
        print("Saving models...")
        
        # Save DataFrame
        df_path = self.output_dir / "df.pkl"
        with open(df_path, "wb") as f:
            pickle.dump(self.df, f)
        print(f" DataFrame saved to {df_path}")
        
        # Save embeddings
        embeddings_path = self.output_dir / "music_embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        print(f" Embeddings saved to {embeddings_path}")
        
        # Save FAISS index
        index_path = self.output_dir / "music_faiss.index"
        faiss.write_index(self.faiss_index, str(index_path))
        print(f" FAISS index saved to {index_path}")
    
    def train(self):
        """Run complete training pipeline"""
        print("=" * 50)
        print("Starting Music Recommendation Model Training")
        print("=" * 50)
        
        self.load_and_clean_data()
        self.create_embeddings()
        self.build_faiss_index()
        self.save_models()
        
        print("=" * 50)
        print("Training completed successfully!")
        print("=" * 50)
