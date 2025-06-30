#!/usr/bin/env python3
"""
Email preprocessing utilities for spam/phishing detection
"""

import re
import string
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler


class EmailPreprocessor:
    """Preprocess emails for ML models"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.count_vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def clean_text(self, text: str) -> str:
        """Clean email text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', ' NUMBER ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, email_content: str, subject: str, sender: str) -> Dict[str, float]:
        """Extract manual features from email"""
        full_text = f"{subject} {email_content}"
        
        features = {
            # Length features
            'content_length': len(email_content),
            'subject_length': len(subject),
            'word_count': len(full_text.split()),
            
            # Character features
            'exclamation_count': full_text.count('!'),
            'question_count': full_text.count('?'),
            'uppercase_ratio': sum(1 for c in full_text if c.isupper()) / max(len(full_text), 1),
            
            # URL features
            'url_count': len(re.findall(r'http[s]?://', full_text)),
            'short_url_count': len(re.findall(r'(bit\.ly|tinyurl|short\.link)', full_text.lower())),
            
            # Suspicious patterns
            'money_symbols': len(re.findall(r'[$£€¥]', full_text)),
            'percentage_symbols': full_text.count('%'),
            
            # Keywords
            'urgent_words': len(re.findall(r'\b(urgent|immediate|act now|expire|suspend)\b', full_text.lower())),
            'winner_words': len(re.findall(r'\b(winner|congratulations|prize|lottery|selected)\b', full_text.lower())),
            'verify_words': len(re.findall(r'\b(verify|confirm|update|validate|click here)\b', full_text.lower())),
            
            # Sender features
            'sender_has_numbers': 1 if any(c.isdigit() for c in sender) else 0,
            'sender_domain_suspicious': 1 if re.search(r'\.(tk|ml|ga|cf)$', sender) else 0,
        }
        
        return features
    
    def fit(self, emails: List[Tuple[str, str, str]], labels: List[int]):
        """Fit the vectorizers on training data"""
        # Combine all text
        all_text = []
        for content, subject, sender in emails:
            cleaned_text = self.clean_text(f"{subject} {content}")
            all_text.append(cleaned_text)
        
        # Fit vectorizers
        self.tfidf_vectorizer.fit(all_text)
        self.count_vectorizer.fit(all_text)
        
        # Extract manual features for scaling
        manual_features = []
        for content, subject, sender in emails:
            features = self.extract_features(content, subject, sender)
            manual_features.append(list(features.values()))
        
        # Fit scaler
        self.scaler.fit(manual_features)
        
        self.is_fitted = True
        
    def transform(self, email_content: str, subject: str, sender: str) -> Dict[str, np.ndarray]:
        """Transform email to feature vectors"""
        if not self.is_fitted:
            # Use default transformation if not fitted
            cleaned_text = self.clean_text(f"{subject} {email_content}")
            
            # Create dummy vectors
            tfidf_features = np.zeros(self.max_features)
            count_features = np.zeros(self.max_features)
            
            # Extract manual features
            manual_features = self.extract_features(email_content, subject, sender)
            manual_array = np.array(list(manual_features.values()))
            
            # Simple normalization if not fitted
            manual_array = (manual_array - manual_array.mean()) / (manual_array.std() + 1e-8)
        else:
            # Clean text
            cleaned_text = self.clean_text(f"{subject} {email_content}")
            
            # Get TF-IDF features
            tfidf_features = self.tfidf_vectorizer.transform([cleaned_text]).toarray()[0]
            
            # Get count features
            count_features = self.count_vectorizer.transform([cleaned_text]).toarray()[0]
            
            # Get manual features
            manual_features = self.extract_features(email_content, subject, sender)
            manual_array = self.scaler.transform([list(manual_features.values())])[0]
        
        return {
            'tfidf': tfidf_features,
            'count': count_features,
            'manual': manual_array,
            'combined': np.concatenate([tfidf_features, manual_array])
        }
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names for interpretation"""
        manual_names = [
            'content_length', 'subject_length', 'word_count',
            'exclamation_count', 'question_count', 'uppercase_ratio',
            'url_count', 'short_url_count', 'money_symbols',
            'percentage_symbols', 'urgent_words', 'winner_words',
            'verify_words', 'sender_has_numbers', 'sender_domain_suspicious'
        ]
        
        if self.is_fitted:
            return {
                'tfidf': self.tfidf_vectorizer.get_feature_names_out().tolist(),
                'manual': manual_names
            }
        else:
            return {'manual': manual_names}


# Shared instance for demo purposes
preprocessor = EmailPreprocessor()