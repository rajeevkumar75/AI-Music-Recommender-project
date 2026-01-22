import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from typing import Optional

# Download required NLTK resources:
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataProcessor:
    """Handles data loading, cleaning, and text preprocessing"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
    
    def load_data(self, csv_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            csv_path: Path to CSV file
            sample_size: Optional sample size for testing/demo
            
        Returns:
            DataFrame with data
        """
        df = pd.read_csv(csv_path)
        df = df.sample(15000)
        
        # Remove unnecessary columns
        if 'link' in df.columns:
            df = df.drop(columns=['link'])
        
        # Sample if specified
        if sample_size:
            df = df.sample(sample_size, random_state=42)
        
        return df.reset_index(drop=True)
    
    def check_data_quality(self, df: pd.DataFrame):
        """Print data quality metrics"""
        print(f"Shape: {df.shape}")
        print(f"Null values:\n{df.isnull().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        return text.str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex=True)
    
    def tokenize_and_stem(self, text: str) -> str:
        """
        Tokenize and apply stemming to text
        
        Args:
            text: Text to process
            
        Returns:
            Processed text
        """
        tokens = wordpunct_tokenize(text)
        stemmed = [self.stemmer.stem(word) for word in tokens]
        return " ".join(stemmed)
    
    def preprocess_text_column(self, df: pd.DataFrame, column: str = 'text') -> pd.DataFrame:
        """
        Preprocess text column
        
        Args:
            df: DataFrame
            column: Column name to preprocess
            
        Returns:
            DataFrame with preprocessed text
        """
        df[column] = df[column].str.lower()
        df[column] = df[column].str.replace(r'^\w\s', ' ', regex=True)
        df[column] = df[column].str.replace(r'\n', ' ', regex=True)
        df[column] = df[column].apply(self.tokenize_and_stem)
        
        return df
