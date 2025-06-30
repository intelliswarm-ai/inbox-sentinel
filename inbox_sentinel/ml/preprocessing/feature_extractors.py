"""
Feature extraction modules for email analysis
"""

import re
from typing import Dict, List
from abc import ABC, abstractmethod
import numpy as np


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    @abstractmethod
    def extract(self, email_content: str, subject: str, sender: str) -> Dict[str, float]:
        """Extract features from email components"""
        pass


class URLFeatureExtractor(BaseFeatureExtractor):
    """Extract URL-related features"""
    
    def __init__(self):
        self.url_shorteners = ['bit.ly', 'tinyurl', 'short.link', 'goo.gl', 'ow.ly']
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        
    def extract(self, email_content: str, subject: str, sender: str) -> Dict[str, float]:
        """Extract URL features"""
        full_text = f"{subject} {email_content}"
        
        features = {
            'url_count': len(re.findall(r'https?://[^\s]+', full_text)),
            'short_url_count': sum(1 for shortener in self.url_shorteners 
                                 if shortener in full_text.lower()),
            'ip_url_count': len(re.findall(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', full_text)),
            'suspicious_tld_count': sum(1 for tld in self.suspicious_tlds 
                                      if tld in full_text.lower()),
        }
        
        return features


class ContentFeatureExtractor(BaseFeatureExtractor):
    """Extract content-based features"""
    
    def __init__(self):
        self.urgent_words = [
            'urgent', 'immediate', 'act now', 'expire', 'suspend',
            'limited time', 'hurry', 'deadline', 'final notice'
        ]
        self.money_words = [
            'winner', 'congratulations', 'prize', 'lottery', 'million',
            'thousand', 'dollars', 'pounds', 'euros', 'cash', 'money'
        ]
        self.action_words = [
            'verify', 'confirm', 'update', 'validate', 'click here',
            'follow link', 'open attachment', 'download', 'install'
        ]
        
    def extract(self, email_content: str, subject: str, sender: str) -> Dict[str, float]:
        """Extract content features"""
        full_text = f"{subject} {email_content}"
        text_lower = full_text.lower()
        
        features = {
            # Length features
            'content_length': len(email_content),
            'subject_length': len(subject),
            'word_count': len(full_text.split()),
            
            # Character features
            'exclamation_count': full_text.count('!'),
            'question_count': full_text.count('?'),
            'uppercase_ratio': sum(1 for c in full_text if c.isupper()) / max(len(full_text), 1),
            
            # Special symbols
            'money_symbols': len(re.findall(r'[$£€¥₹]', full_text)),
            'percentage_symbols': full_text.count('%'),
            
            # Keywords
            'urgent_words': sum(1 for word in self.urgent_words if word in text_lower),
            'money_words': sum(1 for word in self.money_words if word in text_lower),
            'action_words': sum(1 for word in self.action_words if word in text_lower),
            
            # Text patterns
            'all_caps_words': len(re.findall(r'\b[A-Z]{2,}\b', full_text)),
            'repeated_punctuation': len(re.findall(r'[!?]{2,}', full_text)),
        }
        
        return features


class HeaderFeatureExtractor(BaseFeatureExtractor):
    """Extract header/sender-related features"""
    
    def __init__(self):
        self.suspicious_domains = [
            'tk', 'ml', 'ga', 'cf', 'click', 'download', 'win',
            'prize', 'winner', 'lottery', 'casino', 'pharma'
        ]
        self.legitimate_providers = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            'aol.com', 'icloud.com', 'protonmail.com'
        ]
        
    def extract(self, email_content: str, subject: str, sender: str) -> Dict[str, float]:
        """Extract header features"""
        features = {
            'sender_has_numbers': 1.0 if any(c.isdigit() for c in sender.split('@')[0]) else 0.0,
            'sender_length': len(sender),
            'sender_special_chars': sum(1 for c in sender if not c.isalnum() and c not in '@.-_'),
        }
        
        # Domain analysis
        if '@' in sender:
            domain = sender.split('@')[1].lower()
            username = sender.split('@')[0].lower()
            
            features.update({
                'sender_domain_suspicious': 1.0 if any(susp in domain for susp in self.suspicious_domains) else 0.0,
                'sender_legitimate_provider': 1.0 if domain in self.legitimate_providers else 0.0,
                'sender_subdomain_count': domain.count('.'),
                'sender_domain_length': len(domain),
                'sender_username_length': len(username),
                'sender_noreply': 1.0 if 'no-reply' in username or 'noreply' in username else 0.0,
            })
        else:
            # Invalid sender format
            features.update({
                'sender_domain_suspicious': 1.0,
                'sender_legitimate_provider': 0.0,
                'sender_subdomain_count': 0.0,
                'sender_domain_length': 0.0,
                'sender_username_length': 0.0,
                'sender_noreply': 0.0,
            })
        
        # Subject patterns
        features.update({
            'subject_starts_reply': 1.0 if subject.lower().startswith(('re:', 'fw:', 'fwd:')) else 0.0,
            'subject_all_caps': 1.0 if subject.isupper() and len(subject) > 5 else 0.0,
            'subject_contains_brackets': 1.0 if re.search(r'\[.*\]', subject) else 0.0,
        })
        
        return features