# utils/text_utils.py

import re

def remove_parentheses(text: str) -> str:
    """
    Removes all substrings enclosed in parentheses from the input text.
    
    Example:
      "200Ohm (typical)" -> "200Ohm"
    """
    return re.sub(r'\([^)]*\)', '', text).strip()

def extract_parenthetical_content(text: str) -> list:
    """
    Extracts and returns all substrings enclosed in parentheses (including the parentheses)
    from the input text.
    
    Example:
      "Voltage (nominal) 5V" -> ["(nominal)"]
    """
    return re.findall(r'\([^)]*\)', text)

def normalize_whitespace(text: str) -> str:
    """
    Replaces multiple consecutive whitespace characters with a single space,
    and trims the text.
    
    Example:
      "  This   is   a test  " -> "This is a test"
    """
    return re.sub(r'\s+', ' ', text).strip()

def clean_value_string(text: str) -> str:
    """
    Applies general cleaning to a value string. This includes:
      - Fixing common spacing issues between numbers and units
        (e.g., "200Ohm" becomes "200 Ohm").
      - Normalizing extra whitespace.
    
    Example:
      " 200Ohm  (typ) " -> "200 Ohm (typ)"
    """
    # Insert a space between a digit and a letter if not already present
    cleaned = re.sub(r'(\d)([a-zA-Zµ]+)', r'\1 \2', text)
    return normalize_whitespace(cleaned)
