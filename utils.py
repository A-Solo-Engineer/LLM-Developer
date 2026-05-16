import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
from urllib.parse import urlparse, quote_plus

import requests
from bs4 import BeautifulSoup


def validate_input(text: str, max_size_mb: float = 50) -> Dict[str, Any]:
    """
    Validate user input for safety and format.
    
    Args:
        text: Input text to validate
        max_size_mb: Maximum allowed size in MB
    
    Returns:
        Dictionary with validation result and error message if invalid
    """
    if not text or not isinstance(text, str):
        return {'valid': False, 'error': 'Input text is empty or not a string'}
    
    # Check size
    size_mb = len(text.encode('utf-8')) / (1024 * 1024)
    if size_mb > max_size_mb:
        return {'valid': False, 'error': f'Input size ({size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)'}
    
    # Check for required format
    if 'USER INPUT:' not in text or 'OUTPUT:' not in text:
        return {'valid': False, 'error': 'Input must contain "USER INPUT:" and "OUTPUT:" labels'}
    
    return {'valid': True, 'error': None}


def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing potentially harmful characters.
    
    Args:
        text: Text to sanitize
    
    Returns:
        Sanitized text
    """
    # Remove null bytes and control characters (except newlines, tabs)
    sanitized = ''.join(char for char in text if char.isprintable() or char in '\n\t\r')
    return sanitized


def validate_parsed_data(data: List[Tuple[str, str]], min_samples: int = 1) -> Dict[str, Any]:
    """
    Validate parsed training data.
    
    Args:
        data: List of (input, output) tuples
        min_samples: Minimum required samples
    
    Returns:
        Dictionary with validation result
    """
    if not data or len(data) < min_samples:
        return {'valid': False, 'error': f'Need at least {min_samples} training pairs, got {len(data) if data else 0}'}
    
    # Check for empty strings in pairs
    for i, (inp, out) in enumerate(data):
        if not inp or not out:
            return {'valid': False, 'error': f'Pair {i} has empty input or output'}
    
    return {'valid': True, 'error': None}


def validate_hyperparameters(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate training hyperparameters against safe ranges.
    
    Args:
        hyperparams: Dictionary of hyperparameters to validate
    
    Returns:
        Dictionary with validation result
    """
    from config import Config
    
    checks = [
        ('batch_size', Config.BATCH_SIZE_RANGE),
        ('learning_rate', Config.LEARNING_RATE_RANGE),
        ('epochs', Config.EPOCHS_RANGE),
    ]
    
    for param_name, (min_val, max_val) in checks:
        if param_name in hyperparams:
            val = hyperparams[param_name]
            if not (min_val <= val <= max_val):
                return {'valid': False, 'error': f'{param_name} {val} out of range [{min_val}, {max_val}]'}
    
    return {'valid': True, 'error': None}


def get_current_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return current system date and time."""
    return datetime.now().strftime(format_str)


def is_datetime_question(text: str) -> bool:
    """Detect whether the user is asking for date or time."""
    if not text or not isinstance(text, str):
        return False

    patterns = [
        r"\bwhat(?:'s| is)? the (current )?(date|time)\b",
        r"\b(current|today|now)\b",
        r"\bwhat(?:'s| is)? today's date\b",
        r"\bwhat time is it\b",
        r"\bcurrent date\b",
        r"\bcurrent time\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def fetch_webpage_text(url: str, max_chars: int = 1500) -> str:
    """Fetch and summarize the main text from a web page."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        body_text = " ".join(paragraphs).strip()
        if not body_text:
            body_text = soup.get_text(" ", strip=True)
        result = f"URL: {url}\nTitle: {title}\nContent: {body_text}"
        return result[:max_chars]
    except Exception:
        return ""


def search_web(query: str, max_results: int = 3, max_chars: int = 1500) -> str:
    """Perform a lightweight web search and summarize the top results."""
    try:
        search_url = "https://html.duckduckgo.com/html"
        response = requests.get(search_url, params={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        result_items = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            title = link.get_text(" ", strip=True)
            if href.startswith("http") and "duckduckgo.com" not in href and title:
                result_items.append((title, href))
            if len(result_items) >= max_results:
                break
        if not result_items:
            return ""
        summary = [f"Result {i+1}: {title} ({href})" for i, (title, href) in enumerate(result_items, start=1)]
        return ("Search results:\n" + "\n".join(summary))[:max_chars]
    except Exception:
        return ""


def fetch_web_data(query: str, max_chars: int = 1500) -> str:
    """Fetch page text or perform a lightweight web search."""
    if not query or not isinstance(query, str):
        return ""

    query = query.strip()
    if is_valid_url(query):
        content = fetch_webpage_text(query, max_chars=max_chars)
        return content if content else f"Unable to fetch content from {query}."

    search_result = search_web(query, max_results=3, max_chars=max_chars)
    return search_result if search_result else f"Unable to fetch search results for '{query}'."


def parse_training_data(text):
    """
    Parses text data in the format:
    USER INPUT: <input>
    OUTPUT: <output>
    
    Returns a list of tuples: [(input, output), ...]
    """
    # Normalize newlines
    text = text.replace('\r\n', '\n')
    
    # Split by the "USER INPUT:" delimiter, but keep the content
    # We use a regex to find all occurrences
    pattern = r"USER INPUT:(.*?)(?=USER INPUT:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    data = []
    for match in matches:
        # Now inside each match, look for "OUTPUT:"
        if "OUTPUT:" in match:
            parts = match.split("OUTPUT:")
            if len(parts) >= 2:
                user_input = parts[0].strip()
                output = parts[1].strip()
                if user_input and output:
                    data.append((user_input, output))
                    
    return data

def prepare_corpus(data_pairs):
    """
    Combines pairs into a single text corpus for training a causal LM.
    We format it as:
    <start>USER INPUT: ... OUTPUT: ...<end>
    
    Or simply:
    USER INPUT: ... \nOUTPUT: ... \n
    """
    text = ""
    for inp, out in data_pairs:
        text += f"USER INPUT: {inp}\nOUTPUT: {out}\n\n"
    return text

class SimpleTokenizer:
    def __init__(self, text=None):
        import string
        # Initialize with all printable characters to avoid errors with unseen chars during inference
        self.chars = sorted(list(set(string.printable)))
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = len(self.chars)
        
        if text:
            self.fit(text)

    def fit(self, text):
        # Add any new characters found in the text
        new_chars = set(text)
        current_chars = set(self.chars)
        all_chars = sorted(list(current_chars.union(new_chars)))
        
        self.chars = all_chars
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = len(self.chars)

    def encode(self, s):
        # Handle unknown characters by skipping them or mapping to a default
        # For simplicity, we just skip unknown chars to prevent crashing
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'chars': self.chars, 'stoi': self.stoi, 'itos': self.itos}, f)
    
    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chars = data['chars']
        self.stoi = data['stoi']
        self.itos = data['itos']
        self.vocab_size = len(self.chars)
