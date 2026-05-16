"""
Logging system for LLM Developer 

Provides training progress logging, error/warning tracking, and performance metrics recording.
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path


class FormattedLogger:
    """Custom logger with formatted output for LLM Developer"""
    
    def __init__(self, name: str, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        # File handler (detailed)
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (simple)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


class TrainingMetrics:
    """Track training metrics"""
    
    def __init__(self):
        self.metrics = {
            'iterations': [],
            'losses': [],
            'learning_rates': [],
            'validation_losses': []
        }
    
    def log_iteration(self, iteration: int, loss: float, lr: float = None):
        """Log iteration metrics"""
        self.metrics['iterations'].append(iteration)
        self.metrics['losses'].append(loss)
        if lr is not None:
            self.metrics['learning_rates'].append(lr)
    
    def log_validation(self, validation_loss: float):
        """Log validation metrics"""
        self.metrics['validation_losses'].append(validation_loss)
    
    def get_latest_loss(self) -> Optional[float]:
        """Get latest loss value"""
        if self.metrics['losses']:
            return self.metrics['losses'][-1]
        return None
    
    def get_metrics(self):
        """Get all collected metrics"""
        return self.metrics
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'iterations': [],
            'losses': [],
            'learning_rates': [],
            'validation_losses': []
        }


# Global logger instance
_logger: Optional[FormattedLogger] = None


def get_logger(name: str = 'llm_studio') -> FormattedLogger:
    """Get or create a logger instance"""
    global _logger
    if _logger is None:
        _logger = FormattedLogger(name)
    return _logger


def initialize_logger(name: str = 'llm_studio', log_dir: str = 'logs') -> FormattedLogger:
    """Initialize a new logger instance"""
    global _logger
    _logger = FormattedLogger(name, log_dir)
    return _logger
