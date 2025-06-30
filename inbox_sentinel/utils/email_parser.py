"""
Email parsing utilities for handling forwarded emails
"""

import re
from typing import Dict, Optional, Tuple
from datetime import datetime
from inbox_sentinel.core.types import Email


class ForwardedEmailParser:
    """Parser for extracting original email from forwarded messages"""
    
    # Common forward patterns
    FORWARD_PATTERNS = [
        r'[-\s]*Forwarded message[-\s]*',
        r'[-\s]*Original Message[-\s]*',
        r'[-\s]*Begin forwarded message[-\s]*',
    ]
    
    # Header patterns
    HEADER_PATTERNS = {
        'from': r'^From:\s*(.+?)(?:\n|$)',
        'date': r'^Date:\s*(.+?)(?:\n|$)',
        'subject': r'^Subject:\s*(.+?)(?:\n|$)',
        'to': r'^To:\s*(.+?)(?:\n|$)',
    }
    
    @classmethod
    def parse_forwarded_email(cls, email_text: str) -> Tuple[Email, Dict[str, str]]:
        """
        Parse a forwarded email and extract the original content
        
        Args:
            email_text: The full forwarded email text
            
        Returns:
            Tuple of (Email object, metadata dict)
        """
        # Find the forwarded message separator
        forward_start = cls._find_forward_start(email_text)
        
        if forward_start == -1:
            # Not a forwarded email, treat as regular email
            return cls._parse_as_regular_email(email_text)
        
        # Extract the forwarded part
        forwarded_section = email_text[forward_start:]
        
        # Extract headers
        headers = cls._extract_headers(forwarded_section)
        
        # Extract body (everything after headers)
        body_start = cls._find_body_start(forwarded_section)
        if body_start != -1:
            body = forwarded_section[body_start:].strip()
        else:
            body = forwarded_section.strip()
        
        # Create Email object
        email = Email(
            content=body,
            subject=headers.get('subject', 'No Subject'),
            sender=cls._extract_sender_email(headers.get('from', ''))
        )
        
        # Create metadata
        metadata = {
            'is_forwarded': True,
            'original_from': headers.get('from', ''),
            'original_date': headers.get('date', ''),
            'original_to': headers.get('to', ''),
            'forward_text': email_text[:forward_start].strip()
        }
        
        return email, metadata
    
    @classmethod
    def _find_forward_start(cls, text: str) -> int:
        """Find the start of the forwarded message"""
        for pattern in cls.FORWARD_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.end()
        return -1
    
    @classmethod
    def _extract_headers(cls, text: str) -> Dict[str, str]:
        """Extract email headers from forwarded section"""
        headers = {}
        
        # Split into lines for processing
        lines = text.split('\n')
        
        # Look for headers in the first part of the message
        for key, pattern in cls.HEADER_PATTERNS.items():
            for i, line in enumerate(lines[:20]):  # Check first 20 lines
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    headers[key] = match.group(1).strip()
                    break
        
        return headers
    
    @classmethod
    def _find_body_start(cls, text: str) -> int:
        """Find where the actual email body starts"""
        lines = text.split('\n')
        
        # Look for empty line after headers
        header_found = False
        for i, line in enumerate(lines):
            if any(re.match(pattern, line, re.IGNORECASE) 
                   for pattern in cls.HEADER_PATTERNS.values()):
                header_found = True
            elif header_found and line.strip() == '':
                # Found empty line after headers
                return sum(len(l) + 1 for l in lines[:i+1])
        
        return -1
    
    @classmethod
    def _extract_sender_email(cls, from_field: str) -> str:
        """Extract email address from From field"""
        # Look for email in angle brackets
        match = re.search(r'<([^>]+@[^>]+)>', from_field)
        if match:
            return match.group(1)
        
        # Look for plain email
        match = re.search(r'([^\s]+@[^\s]+)', from_field)
        if match:
            return match.group(1)
        
        return from_field
    
    @classmethod
    def _parse_as_regular_email(cls, email_text: str) -> Tuple[Email, Dict[str, str]]:
        """Parse as a regular email when no forward markers found"""
        # Simple parsing - treat first line as subject if short
        lines = email_text.strip().split('\n')
        
        if lines and len(lines[0]) < 100:
            subject = lines[0]
            content = '\n'.join(lines[1:]).strip()
        else:
            subject = "No Subject"
            content = email_text.strip()
        
        email = Email(
            content=content,
            subject=subject,
            sender="unknown@example.com"
        )
        
        metadata = {
            'is_forwarded': False,
            'original_text': email_text
        }
        
        return email, metadata


def parse_gmail_forward(email_text: str) -> Email:
    """
    Convenience function to parse Gmail forwarded emails
    
    Args:
        email_text: The forwarded email text
        
    Returns:
        Email object with extracted content
    """
    email, _ = ForwardedEmailParser.parse_forwarded_email(email_text)
    return email