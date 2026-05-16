# LLM Developer Studio v1.0.0 (Beta)

A lightweight software to train, finetune, and test a small Language Model (LLM) on your local CPU. No heavy GPUs required!

> **Note**: This is the beta version of LLM Developer Studio. The production-grade version will be released on June 17th, 2026.

## Prerequisites

- Python 3.8 or higher
- Internet connection for web scraping features (optional)
- At least 4GB RAM (recommended for smooth training)

## Installation Guide

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. For development dependencies:

   ```bash
   pip install -r dev-requirements.txt
   ```

> **Hardware Compatibility**: The software is designed to work on any hardware. Prebuilt models are local Hugging Face transformer directories, so no llama-cpp dependency is required.

## How it Works

LLM Developer Studio provides a complete pipeline for language model development:

1. **Data Preparation**: Parse and tokenize custom training data
2. **Model Training**: Train a GPT-based language model with adjustable hyperparameters
3. **Inference**: Test the trained model or use prebuilt models for generation
4. **Model Management**: Save, load, and manage trained models

The application uses Streamlit for the web interface and PyTorch for model operations.

## New File Structure

```
LLM-Developer/
├── app.py                 # Main Streamlit application
├── model.py               # GPT language model implementation
├── train.py               # Training logic and utilities
├── tokenizer.py           # Custom tokenizer implementation
├── utils.py               # Helper functions and utilities
├── prebuilt.py            # Prebuilt model loading and generation
├── config.py              # Configuration settings
├── logger.py              # Logging system
├── requirements.txt       # Python dependencies
├── dev-requirements.txt   # Development dependencies
├── Images/                # Static assets (logos, etc.)
├── logs/                  # Generated log files
├── saved_models/          # Saved trained model directories
├── prebuilt/models/       # Local prebuilt model files
├── tests/                 # Unit tests
└── README.md             # This file
```

## Walkthrough

1. **Launch the Application**:

   ```bash
   streamlit run app.py
   ```

2. **Data Tab**: Input your training data in USER INPUT/OUTPUT format and parse it.

3. **Training Tab**: Configure hyperparameters and start training. Monitor progress with real-time loss graphs.

4. **Inference Tab**: Test your trained model or load prebuilt models like Cedar. Adjust generation parameters for different outputs.

5. **Model Info Tab**: Save trained models, view training statistics, and manage model files.

## Basics of Training and Inference of an LM

### Training Concepts

- **Learning Rate**: Controls how much the model adjusts during training. Higher values learn faster but may be unstable.
- **Batch Size**: Number of training examples processed together. Larger batches use more memory but train faster.
- **Epochs**: Complete passes through the training data. More epochs can improve performance but risk overfitting.
- **Validation**: Monitors model performance on unseen data to prevent overfitting.

### Inference Parameters

- **Temperature**: Controls randomness in generation. Lower values (0.1-0.5) make output more focused and deterministic. Higher values (0.7-1.0) increase creativity and diversity.
- **Top-K**: Limits generation to the K most likely tokens. Reduces unlikely outputs, making text more coherent.
- **Top-P (Nucleus Sampling)**: Considers tokens until their cumulative probability reaches P. Provides more dynamic control than Top-K.
- **Max New Tokens**: Maximum length of generated text.
- **Repetition Penalty**: Reduces likelihood of repeating phrases. Higher values discourage repetition.

### Other Features

- **Early Stopping**: Automatically stops training when validation loss stops improving.
- **Mixed Precision**: Uses float16 for faster training on compatible hardware.
- **Checkpointing**: Saves model state during training for recovery.

## Backend Upgrades

- Enhanced model architecture with improved attention mechanisms
- Optimized training loop with better memory management
- Added support for custom tokenizers and preprocessing
- Improved error handling and logging throughout the pipeline
- Integration with Hugging Face transformers for prebuilt models

## Frontend Updates

- Complete UI redesign with dark theme and modern styling
- Real-time training progress visualization with loss/accuracy graphs
- Improved sidebar with model status and saved model management
- Enhanced parameter controls with tooltips and validation
- Responsive design for better usability across devices

## New Features

- **Prebuilt Model Support**: Load and use local prebuilt models (Cedar, Talkative Dumbo)
- **Web Scraping**: Optional web context for enhanced inference
- **Model Saving/Loading**: Persistent storage of trained models
- **Advanced Inference Controls**: Comprehensive parameter tuning
- **Training Metrics**: Detailed statistics and visualization
- **System Prompts**: Customizable prompts for prebuilt models
- **Date/Time Queries**: Automatic handling of temporal questions

## Contributing

This is a beta release. Feedback and contributions are welcome! Please report issues or submit pull requests.

---

Built using Streamlit, PyTorch, and Hugging Face Transformers.

This software implements a **NanoGPT** (a small Transformer-based language model) from scratch. It treats your input data as a continuous stream of text and learns to predict the next character. It uses a custom character-level tokenizer for simplicity.

## File Structure

- `app.py`: The main Streamlit interface.
- `model.py`: The GPT model architecture (PyTorch).
- `train.py`: Training loop and logic.
- `utils.py`: Data parsing and tokenization utilities.

## A Note For Testers And Developers
The dev-requirements.txt file contains all the major python pips and packages for testing the software. You can also contact me on email: admin.forestritium@gmail.com

---

Author: Aarav [A-Solo-Engineer] | Forestritium
Email: admin.forestritium@gmail.com
Last Updated On: 14th May 2026

---

# PRODUCTION GRADE [v1.0.0] VERSION COMING SOON!
