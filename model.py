"""
Enhanced GPT Language Model with architecture improvements for Phase 2

Features:
- RMSNorm as alternative to LayerNorm (faster on CPU)
- Configurable embedding dimensions (32-512)
- Configurable number of layers (1-12)
- Better weight initialization strategies
- Improved dropout implementation
- Model versioning and metadata
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from typing import Optional, Dict, Any


def safe_load_model_state(model, state_dict_path: str, device: str = 'cpu'):
    """
    Safely load model state from file with error handling.
    
    Args:
        model: Model to load state into
        state_dict_path: Path to state dict file
        device: Device to load on
    
    Returns:
        model with loaded state
    
    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If state dict is incompatible
    """
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Model file not found: {state_dict_path}")
    
    try:
        state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model state: {e}")


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - faster alternative to LayerNorm.
    Particularly efficient on CPU. Inspired by T5.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention"""
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        # Compute attention scores with scaled dot-product
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # Apply attention to values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head attention forward pass"""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward network: Linear -> Activation -> Linear -> Dropout"""

    def __init__(self, n_embd: int, dropout: float = 0.0, expansion_factor: float = 4.0):
        super().__init__()
        hidden_size = int(n_embd * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.GELU(),  # Better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block: Multi-head attention followed by feed-forward.
    Includes layer normalization and residual connections.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        use_rmsnorm: bool = False,
        expansion_factor: float = 4.0,
    ):
        super().__init__()
        head_size = n_embd // n_head
        
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout, expansion_factor)
        
        # Choose normalization layer
        if use_rmsnorm:
            self.ln1 = RMSNorm(n_embd)
            self.ln2 = RMSNorm(n_embd)
        else:
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer block forward pass with residual connections"""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Enhanced GPT Language Model with configurable architecture.
    
    Supports:
    - Configurable embedding dimensions (16-512)
    - Configurable number of layers (1-12)
    - Configurable attention heads
    - RMSNorm or LayerNorm
    - Improved weight initialization
    - Model versioning
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 128,
        dropout: float = 0.1,
        use_rmsnorm: bool = False,
        expansion_factor: float = 4.0,
        init_std: float = 0.02,
        model_version: str = "2.0",
    ):
        """
        Initialize GPT Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            n_embd: Embedding dimension (16-512)
            n_head: Number of attention heads (must divide n_embd)
            n_layer: Number of transformer blocks (1-12)
            block_size: Context window size
            dropout: Dropout probability
            use_rmsnorm: Use RMSNorm instead of LayerNorm (faster on CPU)
            expansion_factor: Feed-forward expansion factor
            init_std: Standard deviation for weight initialization
            model_version: Version string for model tracking
        """
        super().__init__()
        
        # Validate configuration
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        assert 16 <= n_embd <= 512, f"n_embd must be between 16 and 512, got {n_embd}"
        assert 1 <= n_layer <= 12, f"n_layer must be between 1 and 12, got {n_layer}"
        
        # Model configuration
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.model_version = model_version
        self.init_std = init_std
        
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd,
                    n_head,
                    block_size,
                    dropout,
                    use_rmsnorm=use_rmsnorm,
                    expansion_factor=expansion_factor,
                )
                for _ in range(n_layer)
            ]
        )
        
        # Final layer normalization
        if use_rmsnorm:
            self.ln_f = RMSNorm(n_embd)
        else:
            self.ln_f = nn.LayerNorm(n_embd)
        
        # Language model head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with improved strategy"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def load_safe(self, state_dict_path: str, device: str = 'cpu'):
        """
        Safely load model state from file.
        
        Args:
            state_dict_path: Path to model state dict
            device: Device to load on
        
        Returns:
            self for chaining
        """
        return safe_load_model_state(self, state_dict_path, device)

    def save_safe(self, save_path: str) -> str:
        """
        Safely save model state to file.
        
        Args:
            save_path: Path to save model to
        
        Returns:
            Path where model was saved
        """
        try:
            torch.save(self.state_dict(), save_path)
            return save_path
        except Exception as e:
            raise RuntimeError(f"Error saving model: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving/loading"""
        return {
            'vocab_size': self.vocab_size,
            'n_embd': self.n_embd,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'block_size': self.block_size,
            'model_version': self.model_version,
        }

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass through model.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for training (B, T)
        
        Returns:
            (logits, loss) tuple. loss is None if targets not provided.
        """
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.embedding_dropout(x)
        
        # Transformer blocks
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter (0-1)
            top_k: Top-k sampling parameter (generate from top k most likely)
            repetition_penalty: Penalty to apply to repeated tokens.
            do_sample: Whether to sample tokens or use greedy decoding.
            eos_token_id: Optional token id to stop generation early.
        
        Returns:
            Extended token sequence
        """
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                token_ids = idx_cond.squeeze(0).tolist() if idx_cond.dim() == 2 else idx_cond.tolist()
                for token_id in set(token_ids):
                    logits[:, token_id] = logits[:, token_id] / repetition_penalty
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Top-K sampling
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(probs, top_k)
                probs_topk = torch.zeros_like(probs)
                probs_topk.scatter_(1, indices, values)
                probs = probs_topk / probs_topk.sum(dim=-1, keepdim=True)
            
            # Nucleus (top-p) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample or greedy selection
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        
        return idx
