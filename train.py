import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from model import GPTLanguageModel
from utils import SimpleTokenizer, validate_hyperparameters, validate_parsed_data
from logger import get_logger, TrainingMetrics
from config import Config
import os
import json
from typing import Optional, Tuple, Callable, Dict, Any

logger = get_logger('training')
metrics = TrainingMetrics()


def validate_training_input(text_data: str, hyperparameters: dict) -> tuple:
    """
    Validate training input data and hyperparameters.
    
    Returns:
        (is_valid, error_message)
    """
    # Validate data
    if not text_data or len(text_data.strip()) == 0:
        return False, "Training data is empty"
    
    # Validate hyperparameters
    hp_validation = validate_hyperparameters(hyperparameters)
    if not hp_validation['valid']:
        return False, f"Invalid hyperparameters: {hp_validation['error']}"
    
    return True, None


def get_batch(data, block_size, batch_size, device):
    """Get a random batch of data safely"""
    if len(data) <= block_size:
        raise ValueError(f"Data length ({len(data)}) must be greater than block_size ({block_size})")
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device, use_mixed_precision: bool = False):
    """Estimate validation loss"""
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        try:
            X, Y = get_batch(data, block_size, batch_size, device)
            with autocast(enabled=use_mixed_precision):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        except Exception as e:
            logger.warning(f"Error during validation iteration {k}: {e}")
            losses[k] = float('inf')
    out = losses.mean()
    model.train()
    return out


class CosineAnnealingScheduler:
    """Cosine annealing learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, base_lr: float, total_steps: int, warmup_steps: int = 0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        """Adjust learning rate"""
        if self.current_step < self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = 0.5 * self.base_lr * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
        return lr


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.
        Returns True if should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (best: {self.best_epoch})")
                return True
        return False


def save_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[CosineAnnealingScheduler],
    epoch: int,
    loss: float,
    checkpoint_path: str = "checkpoint.pt",
    scaler: Optional[GradScaler] = None,
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = {
            'current_step': scheduler.current_step,
            'base_lr': scheduler.base_lr,
            'total_steps': scheduler.total_steps,
            'warmup_steps': scheduler.warmup_steps,
        }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str = "checkpoint.pt",
    device: str = 'cpu'
) -> Tuple[int, float]:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    return epoch, loss


def save_model_safely(
    model: GPTLanguageModel,
    tokenizer: SimpleTokenizer,
    model_path: str = 'model.pt',
    tokenizer_path: str = 'tokenizer.pkl',
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model and tokenizer with error handling and checksums.
    
    Args:
        model: GPT model to save
        tokenizer: Tokenizer to save
        model_path: Path to save model
        tokenizer_path: Path to save tokenizer
        metadata: Optional metadata dictionary
    """
    try:
        # Save model
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save tokenizer
        tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def train_model(
    text_data: str,
    hyperparameters: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
    use_lr_scheduling: bool = True,
    use_gradient_clipping: bool = True,
    use_early_stopping: bool = False,
    enable_checkpointing: bool = False,
    device: Optional[str] = None,
    use_mixed_precision: bool = False,
) -> Tuple[GPTLanguageModel, SimpleTokenizer, float]:
    """
    Train GPT model with Phase 2 optimizations.
    
    Args:
        text_data: Training corpus
        hyperparameters: Dict with batch_size, block_size, epochs, learning_rate
        progress_callback: Optional callback for progress updates
        use_lr_scheduling: Whether to use cosine annealing + warmup
        use_gradient_clipping: Whether to clip gradients
        use_early_stopping: Whether to use early stopping
        enable_checkpointing: Whether to save checkpoints
    
    Returns:
        (model, tokenizer, final_loss)
    
    Raises:
        ValueError: If input validation fails
        RuntimeError: If training encounters unrecoverable errors
    """
    try:
        # Validate input
        logger.info("Validating training input...")
        is_valid, error_msg = validate_training_input(text_data, hyperparameters)
        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            raise ValueError(f"Input validation failed: {error_msg}")
        
        # 1. Tokenize
        logger.info("Tokenizing data...")
        tokenizer = SimpleTokenizer(text_data)
        data = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)
        
        if len(data) == 0:
            raise ValueError("Tokenized data is empty")
        
        # Split into train/val
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}")
        
        vocab_size = tokenizer.vocab_size
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Unpack hyperparameters
        batch_size = int(hyperparameters.get('batch_size', 32))
        block_size = int(hyperparameters.get('block_size', 64))
        learning_rate = float(hyperparameters.get('learning_rate', 1e-3))
        epochs = int(hyperparameters.get('epochs', 10))
        
        # Get model config from hyperparameters or use defaults
        n_embd = int(hyperparameters.get('n_embd', 64))
        n_head = int(hyperparameters.get('n_head', 4))
        n_layer = int(hyperparameters.get('n_layer', 4))
        dropout = float(hyperparameters.get('dropout', 0.1))
        use_rmsnorm = hyperparameters.get('use_rmsnorm', False)
        
        # Detect device or accept override
        requested_device = device if device is not None else Config.detect_device()
        if requested_device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            requested_device = 'cpu'
        device = torch.device(requested_device)
        logger.info(f"Using device: {device}")

        if use_mixed_precision and device.type != 'cuda':
            logger.warning("Mixed precision requested but CUDA is not available; disabling AMP")
            use_mixed_precision = False

        scaler = GradScaler() if use_mixed_precision else None
        
        # Calculate iterations
        iter_per_epoch = max(1, len(train_data) // batch_size)
        max_iters = epochs * iter_per_epoch
        logger.info(f"Training for {epochs} epochs ({max_iters} iterations)")
        
        # Create model
        logger.info("Creating model...")
        model = GPTLanguageModel(
            vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=block_size,
            dropout=dropout,
            use_rmsnorm=use_rmsnorm,
        )
        model = model.to(device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        logger.info(f"Model config: emb={n_embd}, heads={n_head}, layers={n_layer}, dropout={dropout}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        logger.info(f"Optimizer: AdamW with lr={learning_rate}")
        
        # Learning rate scheduling
        scheduler = None
        if use_lr_scheduling:
            warmup_steps = max(1, max_iters // 20)  # 5% warmup
            scheduler = CosineAnnealingScheduler(optimizer, learning_rate, max_iters, warmup_steps)
            logger.info(f"LR Scheduler: Cosine annealing with {warmup_steps} warmup steps")
        
        # Early stopping
        early_stopping = None
        if use_early_stopping:
            early_stopping = EarlyStopping(patience=5, min_delta=1e-3)
            logger.info("Early stopping enabled")
        
        train_loss = 0.0
        best_val_loss = float('inf')
        training_loss_history = []
        val_loss_history = []
        accuracy_history = []
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            
            for iter_in_epoch in range(iter_per_epoch):
                total_iter = epoch * iter_per_epoch + iter_in_epoch
                
                try:
                    # Sample batch
                    xb, yb = get_batch(train_data, block_size, batch_size, device)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    with autocast(enabled=use_mixed_precision):
                        logits, loss = model(xb, yb)
                    
                    if use_mixed_precision:
                        scaler.scale(loss).backward()
                        if use_gradient_clipping:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if use_gradient_clipping:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    # Update learning rate (if scheduling)
                    if scheduler is not None:
                        scheduler.step()
                    
                    train_loss = loss.item()
                    epoch_losses.append(train_loss)
                    metrics.log_iteration(total_iter, train_loss, learning_rate)
                    
                    # Progress callback
                    if progress_callback and total_iter % 10 == 0:
                        try:
                            progress_callback(total_iter, max_iters, train_loss)
                        except Exception as e:
                            logger.warning(f"Error in progress callback: {e}")
                            
                except Exception as e:
                    logger.error(f"Error at iteration {total_iter}: {e}")
                    if total_iter < 10:
                        raise RuntimeError(f"Training failed at iteration {total_iter}: {e}")
                    else:
                        logger.warning("Continuing training despite error...")
            
            # Validation after each epoch
            avg_epoch_loss = np.mean(epoch_losses)
            training_loss_history.append(avg_epoch_loss)
            
            if len(val_data) > block_size:
                with torch.no_grad():
                    val_loss = estimate_loss(
                        model,
                        val_data,
                        block_size,
                        batch_size,
                        50,
                        device,
                        use_mixed_precision
                    )
                    val_loss_history.append(val_loss)
                    
                    # Calculate accuracy on validation set
                    model.eval()
                    total_correct = 0
                    total_tokens = 0
                    num_accuracy_batches = 10
                    for _ in range(num_accuracy_batches):
                        xb_acc, yb_acc = get_batch(val_data, block_size, batch_size, device)
                        with autocast(enabled=use_mixed_precision):
                            logits_acc, _ = model(xb_acc, yb_acc)
                        
                        # Get predicted tokens (argmax of logits)
                        predictions = torch.argmax(logits_acc, dim=-1)
                        # Flatten and compare with targets
                        total_correct += (predictions.flatten() == yb_acc.flatten()).sum().item()
                        total_tokens += yb_acc.numel()
                    
                    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
                    accuracy_history.append(accuracy)
                    model.train()
                    
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
                    
                    # Track best validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logger.info(f"✅ New best validation loss: {best_val_loss:.4f}")
                    
                    # Early stopping check
                    if use_early_stopping and early_stopping(val_loss, epoch):
                        logger.info(f"Stopping training at epoch {epoch+1}")
                        break
                    
                    # Checkpointing
                    if enable_checkpointing and (epoch + 1) % 5 == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            val_loss,
                            f"checkpoint_epoch_{epoch+1}.pt",
                            scaler=scaler
                        )
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}")
        
        logger.info(f"Training complete. Final loss: {train_loss:.4f}, Best val loss: {best_val_loss:.4f}")
        
        # Save model
        metadata = {
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'n_params': n_params,
            'epochs': epochs,
            'final_loss': train_loss,
            'best_val_loss': best_val_loss,
            'device': str(device),
            'mixed_precision': use_mixed_precision,
            'training_loss_history': training_loss_history,
            'val_loss_history': val_loss_history,
            'accuracy_history': accuracy_history,
            'optimizations': {
                'lr_scheduling': use_lr_scheduling,
                'gradient_clipping': use_gradient_clipping,
                'early_stopping': use_early_stopping,
            }
        }
        save_model_safely(model, tokenizer, metadata=metadata)
        
        return model, tokenizer, train_loss, training_loss_history, val_loss_history, accuracy_history
        
    except Exception as e:
        logger.critical(f"Training failed: {e}")
        raise


