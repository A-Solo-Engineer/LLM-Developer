# Local LLM Training Studio

A lightweight software to train, finetune, and test a small Language Model (LLM) on your local CPU. No heavy GPUs required!

## Features

- **Custom Data Training**: Input your own `USER INPUT` / `OUTPUT` pairs.
- **Adjustable Hyperparameters**: Control Learning Rate, Batch Size, and Epochs.
- **Inference Playground**: Test your model with adjustable Temperature and Nucleus Sampling (Top-P).
- **Lightweight**: Uses a miniature GPT architecture implemented in pure PyTorch (no Hugging Face `transformers` dependency).
- **CPU Friendly**: Designed to run on standard laptops.

## Prerequisites

- Python 3.8 or higher

## Installation

1.  Open a terminal in this folder.
2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the application:

    ```bash
    streamlit run app.py
    ```

2.  **Tab 1 (Data)**:
    - Paste your training data in the text box.
    - Click **Parse & Preview** to prepare the data.

3.  **Tab 2 (Training)**:
    - Set your desired parameters.
    - Click **Start Training**. Watch the loss go down!

4.  **Tab 3 (Inference)**:
    - Once training is complete, switch to this tab.
    - Type a message and see how your model responds.
    - Play with **Temperature** and **Top-P** to change the creativity.

## How it works

This software implements a **NanoGPT** (a small Transformer-based language model) from scratch. It treats your input data as a continuous stream of text and learns to predict the next character. It uses a custom character-level tokenizer for simplicity.

## File Structure

- `app.py`: The main Streamlit interface.
- `model.py`: The GPT model architecture (PyTorch).
- `train.py`: Training loop and logic.
- `utils.py`: Data parsing and tokenization utilities.
