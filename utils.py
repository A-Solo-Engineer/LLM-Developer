import re

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
