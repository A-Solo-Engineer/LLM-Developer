"""
Improved Tokenizer with Subword Tokenization (BPE-like)

Replaces character-level tokenizer with subword tokenization for better vocabulary efficiency
and larger effective vocabulary with same memory footprint.
"""

import json
import pickle
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter


class SubwordTokenizer:
    """
    Byte Pair Encoding (BPE)-inspired subword tokenizer.
    More efficient than character-level tokenization for longer sequences.
    """
    
    # Special tokens
    SPECIAL_TOKENS = {
        '<START>': 0,
        '<END>': 1,
        '<PAD>': 2,
        '<UNK>': 3,
    }
    
    def __init__(self, vocab_size: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (includes special tokens)
        """
        self.vocab_size = vocab_size
        self.token_to_id = dict(self.SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merges = []  # Track merges for consistent encoding
        self.unk_id = self.SPECIAL_TOKENS['<UNK>']
        self.pad_id = self.SPECIAL_TOKENS['<PAD>']
        self.start_id = self.SPECIAL_TOKENS['<START>']
        self.end_id = self.SPECIAL_TOKENS['<END>']
    
    def train(self, text: str, num_merges: int = None):
        """
        Train tokenizer on text using BPE algorithm.
        
        Args:
            text: Training text
            num_merges: Number of merge operations (default: vocab_size - len(special_tokens))
        """
        if num_merges is None:
            num_merges = max(1, self.vocab_size - len(self.SPECIAL_TOKENS) - 256)
        
        # Start with byte-level tokens (all unique characters)
        vocab = set(text)
        char_to_id = {ch: len(self.SPECIAL_TOKENS) + i for i, ch in enumerate(sorted(vocab))}
        
        # Update token mappings
        for token, token_id in char_to_id.items():
            if token_id not in self.id_to_token:
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
        
        # Encode text into tokens
        tokens = [char_to_id[ch] for ch in text]
        
        # Perform BPE merges
        for merge_idx in range(num_merges):
            if len(tokens) <= 1:
                break
            
            # Count adjacent pairs
            pairs = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
            
            if not pairs:
                break
            
            # Find most frequent pair
            most_common_pair = max(pairs, key=pairs.get)
            
            # Create new token for this pair
            new_token_id = len(self.SPECIAL_TOKENS) + 256 + merge_idx
            if new_token_id >= self.vocab_size:
                break
            
            new_token = f"<{merge_idx}>"
            self.token_to_id[new_token] = new_token_id
            self.id_to_token[new_token_id] = new_token
            self.merges.append((most_common_pair, new_token_id))
            
            # Merge tokens in sequence
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == most_common_pair:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add START and END tokens
        
        Returns:
            List of token IDs
        """
        # Convert to characters first
        char_ids = []
        for ch in text:
            if ch in self.token_to_id:
                char_ids.append(self.token_to_id[ch])
            else:
                char_ids.append(self.unk_id)
        
        # Apply merges
        for (pair_token1, pair_token2), new_token_id in self.merges:
            new_char_ids = []
            i = 0
            while i < len(char_ids):
                if i < len(char_ids) - 1 and char_ids[i] == pair_token1 and char_ids[i + 1] == pair_token2:
                    new_char_ids.append(new_token_id)
                    i += 2
                else:
                    new_char_ids.append(char_ids[i])
                    i += 1
            char_ids = new_char_ids
        
        # Add special tokens if requested
        if add_special_tokens:
            char_ids = [self.start_id] + char_ids + [self.end_id]
        
        return char_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens if requested
                if skip_special_tokens and token.startswith('<'):
                    continue
                tokens.append(token)
            else:
                tokens.append(self.unk_id)
        
        # Join tokens (remove angle brackets from merge tokens)
        text = ''.join(tokens)
        return text
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'merges': self.merges,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.merges = data['merges']


class SimpleTokenizer:
    """
    Backward-compatible character-level tokenizer for Phase 1 models.
    Used primarily for compatibility with existing v1 models.
    """
    
    def __init__(self, text=None):
        import string
        # Initialize with all printable characters to avoid errors with unseen chars during inference
        self.chars = sorted(list(set(string.printable)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        if text:
            self.fit(text)

    def fit(self, text):
        """Fit tokenizer on text"""
        # Add any new characters found in the text
        new_chars = set(text)
        current_chars = set(self.chars)
        all_chars = sorted(list(current_chars.union(new_chars)))
        
        self.chars = all_chars
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        """Encode text to token IDs"""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        """Decode token IDs to text"""
        return ''.join([self.itos[i] for i in l])
    
    def save(self, path):
        """Save tokenizer to file"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'chars': self.chars, 'stoi': self.stoi, 'itos': self.itos}, f)
    
    def load(self, path):
        """Load tokenizer from file"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chars = data['chars']
        self.stoi = data['stoi']
        self.itos = data['itos']
        self.vocab_size = len(self.chars)
