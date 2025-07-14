#!/usr/bin/env python3
"""
NLP Preprocessor for Fake News Detection System
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import string
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPPreprocessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()
        
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words."""
        tokens = word_tokenize(text)
        
        # Remove stop words and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_linguistic_features(self, text):
        """Extract linguistic features from text."""
        blob = TextBlob(text)
        
        features = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'avg_sentence_length': len(text.split()) / len(sent_tokenize(text)) if sent_tokenize(text) else 0,
            'punctuation_count': sum(1 for char in text if char in string.punctuation),
            'uppercase_count': sum(1 for char in text if char.isupper()),
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        }
        
        return features
    
    def extract_readability_features(self, text):
        """Extract readability features."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'automated_readability_index': 0
            }
        
        # Calculate syllable count (approximation)
        def count_syllables(word):
            vowels = 'aeiouy'
            count = 0
            prev_char_was_vowel = False
            
            for char in word.lower():
                if char in vowels:
                    if not prev_char_was_vowel:
                        count += 1
                    prev_char_was_vowel = True
                else:
                    prev_char_was_vowel = False
            
            # Handle silent e
            if word.endswith('e') and count > 1:
                count -= 1
            
            return max(1, count)
        
        total_syllables = sum(count_syllables(word) for word in words)
        total_words = len(words)
        total_sentences = len(sentences)
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59
        
        # Automated Readability Index
        characters = sum(len(word) for word in words)
        automated_readability_index = (4.71 * (characters / total_words)) + (0.5 * (total_words / total_sentences)) - 21.43
        
        return {
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'automated_readability_index': automated_readability_index
        }
    
    def extract_stylometric_features(self, text):
        """Extract stylometric features."""
        words = word_tokenize(text.lower())
        
        # Function words (common in stylometric analysis)
        function_words = {
            'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
            'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had',
            'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when'
        }
        
        function_word_count = sum(1 for word in words if word in function_words)
        
        # POS tag distribution (simplified)
        pos_tags = nltk.pos_tag(words)
        noun_count = sum(1 for word, pos in pos_tags if pos.startswith('N'))
        verb_count = sum(1 for word, pos in pos_tags if pos.startswith('V'))
        adj_count = sum(1 for word, pos in pos_tags if pos.startswith('J'))
        adv_count = sum(1 for word, pos in pos_tags if pos.startswith('R'))
        
        total_words = len(words)
        
        return {
            'function_word_ratio': function_word_count / total_words if total_words > 0 else 0,
            'noun_ratio': noun_count / total_words if total_words > 0 else 0,
            'verb_ratio': verb_count / total_words if total_words > 0 else 0,
            'adjective_ratio': adj_count / total_words if total_words > 0 else 0,
            'adverb_ratio': adv_count / total_words if total_words > 0 else 0,
            'lexical_diversity': len(set(words)) / len(words) if words else 0
        }
    
    def process_dataset(self, df, text_column='content', target_column='label'):
        """Process entire dataset."""
        logger.info("Starting dataset preprocessing...")
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Extract features
        logger.info("Extracting linguistic features...")
        linguistic_features = df['cleaned_text'].apply(self.extract_linguistic_features)
        linguistic_df = pd.DataFrame(linguistic_features.tolist())
        
        logger.info("Extracting readability features...")
        readability_features = df['cleaned_text'].apply(self.extract_readability_features)
        readability_df = pd.DataFrame(readability_features.tolist())
        
        logger.info("Extracting stylometric features...")
        stylometric_features = df['cleaned_text'].apply(self.extract_stylometric_features)
        stylometric_df = pd.DataFrame(stylometric_features.tolist())
        
        # Combine all features
        feature_df = pd.concat([linguistic_df, readability_df, stylometric_df], axis=1)
        
        # TF-IDF vectorization
        logger.info("Creating TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['cleaned_text'])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine all features
        final_features = pd.concat([feature_df, tfidf_df], axis=1)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(final_features)
        
        logger.info(f"Preprocessing complete. Final feature shape: {scaled_features.shape}")
        
        return scaled_features, final_features.columns.tolist()
    
    def process_single_text(self, text):
        """Process a single text for prediction."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract features
        linguistic_features = self.extract_linguistic_features(cleaned_text)
        readability_features = self.extract_readability_features(cleaned_text)
        stylometric_features = self.extract_stylometric_features(cleaned_text)
        
        # Combine features
        features = {**linguistic_features, **readability_features, **stylometric_features}
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([cleaned_text])
        tfidf_dict = {f'tfidf_{i}': tfidf_features[0, i] for i in range(tfidf_features.shape[1])}
        
        # Combine all features
        all_features = {**features, **tfidf_dict}
        
        # Convert to array and scale
        feature_array = np.array(list(all_features.values())).reshape(1, -1)
        scaled_features = self.scaler.transform(feature_array)
        
        return scaled_features[0]

def main():
    """Main function to demonstrate NLP preprocessing."""
    # Sample data
    sample_data = {
        'content': [
            "This is a real news article with factual information. It contains proper grammar and structure.",
            "SHOCKING!!! You won't believe what happened next! This is totally fake news with lots of exclamation marks!!!",
            "A balanced report on current events with proper citations and sources. The information is well-researched."
        ],
        'label': [1, 0, 1]  # 1 for real, 0 for fake
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize preprocessor
    preprocessor = NLPPreprocessor()
    
    # Process dataset
    features, feature_names = preprocessor.process_dataset(df)
    
    print(f"Feature extraction complete!")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Show some sample features
    print("\nSample linguistic features for first article:")
    sample_features = preprocessor.extract_linguistic_features(df['content'][0])
    for feature, value in sample_features.items():
        print(f"{feature}: {value:.3f}")

if __name__ == "__main__":
    main()