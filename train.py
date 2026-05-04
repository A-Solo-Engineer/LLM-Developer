import torch
import numpy as np
from model import GPTLanguageModel
from utils import SimpleTokenizer
import os

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def train_model(text_data, hyperparameters, progress_callback=None):
    """
    text_data: str, the full training corpus
    hyperparameters: dict
        - batch_size
        - block_size
        - max_iters (calculated from epochs)
        - learning_rate
        - device
        - n_embd, n_head, n_layer
    progress_callback: function(step, loss) -> None
    """
    
    # 1. Tokenize
    tokenizer = SimpleTokenizer(text_data)
    data = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    vocab_size = tokenizer.vocab_size
    
    # Unpack hyperparameters
    batch_size = int(hyperparameters.get('batch_size', 32))
    block_size = int(hyperparameters.get('block_size', 64))
    learning_rate = float(hyperparameters.get('learning_rate', 1e-3))
    epochs = int(hyperparameters.get('epochs', 10))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model config (small default for CPU)
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0
    
    # Calculate iterations based on epochs
    # One epoch = len(train_data) // batch_size
    iter_per_epoch = max(1, len(train_data) // batch_size)
    max_iters = epochs * iter_per_epoch
    
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    m = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_loss = 0
    
    for iter in range(max_iters):
        # sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        if progress_callback and iter % 10 == 0:
            progress_callback(iter, max_iters, train_loss)
            
    # Save model and tokenizer
    torch.save(model.state_dict(), 'model.pt')
    tokenizer.save('tokenizer.pkl')
    
    return model, tokenizer, train_loss
