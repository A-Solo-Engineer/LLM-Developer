"""
Unit tests for LLM Developer Studio

Tests for core functionality including tokenizer, training loop, and model I/O.
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import SimpleTokenizer, parse_training_data, prepare_corpus, validate_input
from model import GPTLanguageModel
from train import train_model, get_batch
from config import Config


class TestInputValidation(unittest.TestCase):
    """Test input validation functions"""
    
    def test_validate_input_valid(self):
        """Test validation with valid input"""
        valid_input = "USER INPUT: hello\nOUTPUT: world"
        result = validate_input(valid_input, max_size_mb=50)
        self.assertTrue(result['valid'])
    
    def test_validate_input_empty(self):
        """Test validation with empty input"""
        result = validate_input("", max_size_mb=50)
        self.assertFalse(result['valid'])
    
    def test_validate_input_too_large(self):
        """Test validation with oversized input"""
        large_input = "x" * (60 * 1024 * 1024)  # 60MB
        result = validate_input(large_input, max_size_mb=50)
        self.assertFalse(result['valid'])
    
    def test_validate_input_missing_format(self):
        """Test validation with missing format"""
        invalid_input = "This doesn't have the right format"
        result = validate_input(invalid_input, max_size_mb=50)
        self.assertFalse(result['valid'])


class TestTokenizer(unittest.TestCase):
    """Test SimpleTokenizer functionality"""
    
    def setUp(self):
        self.text = "USER INPUT: hello\nOUTPUT: world"
        self.tokenizer = SimpleTokenizer(self.text)
    
    def test_encode_decode(self):
        """Test encoding and decoding"""
        encoded = self.tokenizer.encode(self.text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, self.text)
    
    def test_vocab_size(self):
        """Test vocabulary size"""
        self.assertGreater(self.tokenizer.vocab_size, 0)
        self.assertEqual(len(self.tokenizer.stoi), self.tokenizer.vocab_size)
    
    def test_unknown_characters(self):
        """Test handling of unknown characters"""
        text_with_unknown = self.text + "🤖"  # emoji
        encoded = self.tokenizer.encode(text_with_unknown)
        self.assertIsInstance(encoded, list)
        self.assertTrue(len(encoded) > 0)
    
    def test_tokenizer_save_load(self):
        """Test saving and loading tokenizer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'tokenizer.pkl')
            self.tokenizer.save(save_path)
            
            new_tokenizer = SimpleTokenizer()
            new_tokenizer.load(save_path)
            
            self.assertEqual(self.tokenizer.vocab_size, new_tokenizer.vocab_size)
            encoded1 = self.tokenizer.encode(self.text)
            encoded2 = new_tokenizer.encode(self.text)
            self.assertEqual(encoded1, encoded2)


class TestDataParsing(unittest.TestCase):
    """Test data parsing and preparation"""
    
    def test_parse_training_data_valid(self):
        """Test parsing valid training data"""
        data = """USER INPUT: hello
OUTPUT: hi

USER INPUT: how are you
OUTPUT: I'm fine"""
        parsed = parse_training_data(data)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0][0], "hello")
        self.assertEqual(parsed[0][1], "hi")
    
    def test_parse_training_data_empty(self):
        """Test parsing empty data"""
        parsed = parse_training_data("")
        self.assertEqual(len(parsed), 0)
    
    def test_prepare_corpus(self):
        """Test corpus preparation"""
        pairs = [("hello", "hi"), ("how are you", "fine")]
        corpus = prepare_corpus(pairs)
        self.assertIn("USER INPUT:", corpus)
        self.assertIn("OUTPUT:", corpus)
        self.assertGreater(len(corpus), 0)


class TestModelCreation(unittest.TestCase):
    """Test model creation and configuration"""
    
    def test_model_creation(self):
        """Test creating a model"""
        model = GPTLanguageModel(vocab_size=100, n_embd=64, n_head=4, n_layer=2)
        self.assertIsNotNone(model)
    
    def test_model_parameter_count(self):
        """Test parameter counting"""
        model = GPTLanguageModel(vocab_size=100, n_embd=64, n_head=4, n_layer=2)
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 0)
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = GPTLanguageModel(vocab_size=100, n_embd=32, n_head=2, n_layer=2, block_size=64)
        x = torch.randint(0, 100, (2, 64))
        logits, loss = model(x)
        self.assertEqual(logits.shape, (2, 64, 100))
        self.assertIsNone(loss)
    
    def test_model_with_targets(self):
        """Test model with targets (training mode)"""
        model = GPTLanguageModel(vocab_size=100, n_embd=32, n_head=2, n_layer=2, block_size=64)
        x = torch.randint(0, 100, (2, 64))
        y = torch.randint(0, 100, (2, 64))
        logits, loss = model(x, y)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
    
    def test_model_save_load(self):
        """Test saving and loading model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pt')
            
            # Create and save model
            model = GPTLanguageModel(vocab_size=100, n_embd=32, n_head=2, n_layer=2)
            torch.save(model.state_dict(), model_path)
            
            # Load model
            new_model = GPTLanguageModel(vocab_size=100, n_embd=32, n_head=2, n_layer=2)
            new_model.load_state_dict(torch.load(model_path))
            
            # Compare parameters
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


class TestTrainingUtils(unittest.TestCase):
    """Test training utility functions"""
    
    def test_get_batch(self):
        """Test batch creation"""
        data = torch.randint(0, 100, (1000,))
        x, y = get_batch(data, block_size=64, batch_size=32, device='cpu')
        self.assertEqual(x.shape[0], 32)
        self.assertEqual(x.shape[1], 64)
        self.assertEqual(y.shape[0], 32)
        self.assertEqual(y.shape[1], 64)


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def test_get_preset(self):
        """Test getting a preset"""
        preset = Config.get_preset('medium')
        self.assertIsNotNone(preset)
        self.assertEqual(preset.name, 'medium')
    
    def test_list_presets(self):
        """Test listing presets"""
        presets = Config.list_presets()
        self.assertEqual(len(presets), 3)
        self.assertIn('light', presets)
        self.assertIn('medium', presets)
        self.assertIn('heavy', presets)
    
    def test_invalid_preset(self):
        """Test invalid preset"""
        with self.assertRaises(ValueError):
            Config.get_preset('nonexistent')
    
    def test_device_detection(self):
        """Test device detection"""
        device = Config.detect_device()
        self.assertIn(device, ['cuda', 'cpu'])
    
    def test_device_info(self):
        """Test device info retrieval"""
        info = Config.get_device_info()
        self.assertIn('device', info)
        self.assertIn('cuda_available', info)


if __name__ == '__main__':
    unittest.main()
