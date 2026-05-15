import streamlit as st
import torch
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import parse_training_data, prepare_corpus, SimpleTokenizer, validate_input, sanitize_text, get_current_datetime, is_datetime_question, fetch_web_data
from train import train_model
from model import GPTLanguageModel
from prebuilt import load_prebuilt_model, list_prebuilt_models, generate_with_prebuilt_model, get_prebuilt_model_path, PREBUILT_MODEL_ALIASES
from config import Config
from logger import get_logger

# Initialize logger
logger = get_logger('app')

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

# Model download mappings
MODEL_HF_NAMES = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
}

def download_prebuilt_model_if_needed(model_alias: str):
    """Download a prebuilt model if it's not already present locally."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        st.error("❌ Transformers library not installed. Please install requirements.txt")
        return False
    
    model_dir = get_prebuilt_model_path(model_alias)
    if model_dir and os.path.exists(model_dir):
        return True  # Already exists
    
    hf_name = MODEL_HF_NAMES.get(model_alias)
    if not hf_name:
        st.error(f"❌ Unknown model alias: {model_alias}")
        return False
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    with st.spinner(f"Downloading {model_alias} model... This may take several minutes."):
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            tokenizer.save_pretrained(model_dir)
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32)
            model.save_pretrained(model_dir)
            
            st.success(f"✅ Downloaded {model_alias} model successfully!")
            logger.info(f"Downloaded prebuilt model {model_alias} to {model_dir}")
            return True
        except Exception as e:
            st.error(f"❌ Failed to download {model_alias}: {str(e)}")
            logger.error(f"Model download failed for {model_alias}: {e}")
            return False


def ensure_saved_models_dir():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    return SAVED_MODELS_DIR


def list_saved_trained_models():
    saved_dir = ensure_saved_models_dir()
    return sorted(
        [name for name in os.listdir(saved_dir) if os.path.isdir(os.path.join(saved_dir, name))]
    )


def save_trained_model(model_name: str, model: GPTLanguageModel, tokenizer: SimpleTokenizer, metadata: dict):
    if not model_name or len(model_name.strip()) == 0:
        raise ValueError("Model name is required to save a trained model.")
    safe_name = os.path.basename(model_name).strip().replace(" ", "_")
    if len(safe_name) == 0:
        raise ValueError("Invalid model name.")

    save_dir = os.path.join(ensure_saved_models_dir(), safe_name)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.pt")
    tokenizer_path = os.path.join(save_dir, "tokenizer.pkl")
    metadata_path = os.path.join(save_dir, "metadata.json")

    torch.save(model.state_dict(), model_path)
    tokenizer.save(tokenizer_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return save_dir


def load_saved_trained_model(model_name: str):
    if not model_name:
        raise ValueError("No saved model name provided.")
    model_dir = os.path.join(ensure_saved_models_dir(), model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Saved model not found: {model_name}")

    metadata_path = os.path.join(model_dir, "metadata.json")
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    model_path = os.path.join(model_dir, "model.pt")

    if not os.path.exists(metadata_path) or not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Saved model is missing required files.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    config = metadata.get("model_config") or metadata
    model = GPTLanguageModel(
        vocab_size=config["vocab_size"],
        n_embd=config.get("n_embd", 64),
        n_head=config.get("n_head", 4),
        n_layer=config.get("n_layer", 4),
        block_size=config.get("block_size", 128),
        dropout=config.get("dropout", 0.1),
        use_rmsnorm=config.get("use_rmsnorm", False),
        model_version=config.get("model_version", "2.0"),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)

    return model, tokenizer, metadata

st.set_page_config(
    page_title="LLM Developer Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
theme_css = """
<style>
    /* Pure black, high contrast theme */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .css-1r6slb0, .stTabs {
        background-color: rgba(0, 0, 0, 0.98) !important;
        padding: 2rem !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.08) !important;
    }

    h1, h2, h3, h4, h5, h6, .streamlit-expanderHeader {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00ffff, #00d1ff) !important;
        color: #000000 !important;
        border: 1px solid rgba(0, 255, 255, 0.6) !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.4rem !important;
        font-weight: 700 !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.35) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.5) !important;
    }

    .stTextArea textarea, .stTextInput input, textarea, input {
        background-color: #111111 !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    .model-tag {
        display: inline-block;
        background: rgba(0, 255, 255, 0.12) !important;
        color: #00ffff !important;
        padding: 5px 14px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0, 255, 255, 0.28) !important;
        margin: 4px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    .stAlert {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# Sidebar with logo and status
with st.sidebar:
    st.image("Images/Logo.png", width=100)
    st.title("LLM Studio")
    st.markdown("---")
    
    # Here was st.markdown
    
    if 'model' in st.session_state and st.session_state.model:
        st.success("🟢 Model Ready")

    st.markdown("---")
    st.subheader("Saved Trained Models")
    saved_models = list_saved_trained_models()
    if saved_models:
        selected_saved_model = st.selectbox("Load saved model", saved_models, key="saved_model_selector")
        if st.button("Load Saved Model", key="load_saved_model"):
            try:
                model, tokenizer, metadata = load_saved_trained_model(selected_saved_model)
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.training_stats = metadata.get('training_stats', {}) if isinstance(metadata, dict) else {}
                st.success(f"✅ Loaded saved model: {selected_saved_model}")
                logger.info(f"Loaded saved trained model: {selected_saved_model}")
            except Exception as e:
                st.error(f"❌ Could not load saved model: {e}")
                logger.error(f"Saved model load failed: {e}", exc_info=True)
    else:
        st.info("No saved trained models found. Train and save a model to see it here.")

    st.markdown("---")
    st.caption("v1.0.0 | Built by Forestritium")

st.title("🚀 LLM Developer Studio")
st.markdown("### Democratizing Artificial Intelligence")

# Session State for Model and Tokenizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'training_stats' not in st.session_state:
    st.session_state.training_stats = {}
if 'prebuilt_llm' not in st.session_state:
    st.session_state.prebuilt_llm = None
if 'prebuilt_model_path' not in st.session_state:
    st.session_state.prebuilt_model_path = None
if 'prebuilt_loaded' not in st.session_state:
    st.session_state.prebuilt_loaded = False
if 'training_loss_history' not in st.session_state:
    st.session_state.training_loss_history = []
if 'val_loss_history' not in st.session_state:
    st.session_state.val_loss_history = []
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = []

tabs = st.tabs(["1. Data", "2. Training", "3. Inference", "4. Model Info"])

# --- TAB 1: DATA ---
with tabs[0]:
    st.header("Training Data")
    st.info("Format your data as pairs of USER INPUT and OUTPUT.")
    
    default_text = """USER INPUT: Hello
OUTPUT: Hi there! How can I help you today?

USER INPUT: What is this?
OUTPUT: This is a simple LLM training software running on your CPU.

USER INPUT: Who are you?
OUTPUT: I am a small AI model trained by you."""

    data_input = st.text_area("Input Data", value=default_text, height=300)
    
    if st.button("Parse & Preview"):
        try:
            # Validate input
            validation = validate_input(data_input, max_size_mb=Config.MAX_INPUT_SIZE_MB / 1024)
            if not validation['valid']:
                st.error(f"❌ Input Validation Error: {validation['error']}")
                logger.warning(f"Input validation failed: {validation['error']}")
            else:
                # Sanitize input
                sanitized_input = sanitize_text(data_input)
                
                # Parse data
                parsed_data = parse_training_data(sanitized_input)
                
                if len(parsed_data) == 0:
                    st.error("❌ No training pairs found. Please use the format:\nUSER INPUT: ...\nOUTPUT: ...")
                    logger.warning("No training pairs parsed from input")
                else:
                    st.success(f"✅ Successfully parsed {len(parsed_data)} training pairs!")
                    st.write(f"**Sample pairs (first 5):**")
                    for i, (inp, out) in enumerate(parsed_data[:5], 1):
                        st.markdown(f"  {i}. **Input:** {inp[:50]}... → **Output:** {out[:50]}...")
                    
                    st.session_state.parsed_data = parsed_data
                    st.session_state.raw_text = prepare_corpus(parsed_data)
                    logger.info(f"Successfully parsed {len(parsed_data)} training pairs")
        except Exception as e:
            st.error(f"❌ Error parsing data: {str(e)}")
            logger.error(f"Data parsing error: {e}", exc_info=True)

# --- TAB 2: TRAINING ---
with tabs[1]:
    st.header("Training Configuration")
    
    # Preset selector
    preset_col1, preset_col2 = st.columns([2, 1])
    with preset_col1:
        preset_names = list(Config.list_presets().keys())
        selected_preset = st.selectbox(
            "Select Preset",
            preset_names,
            help="Choose a preset configuration or customize below"
        )
    
    with preset_col2:
        show_advanced = st.checkbox("Advanced Mode", help="Customize all parameters")
    
    # Load selected preset values
    selected_config = Config.get_preset(selected_preset)
    
    # Basic parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        lr = st.number_input(
            "Learning Rate",
            value=selected_config.learning_rate,
            format="%.5f",
            min_value=1e-5,
            max_value=0.1,
            disabled=not show_advanced
        )
    with col2:
        batch_size = st.number_input(
            "Batch Size",
            value=selected_config.batch_size,
            min_value=1,
            max_value=256,
            disabled=not show_advanced
        )
    with col3:
        epochs = st.number_input(
            "Epochs",
            value=selected_config.epochs,
            min_value=1,
            max_value=1000,
            disabled=not show_advanced
        )
    
    # Advanced parameters (model architecture)
    if show_advanced:
        st.markdown("**Model Architecture (Phase 2)**")
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        
        with adv_col1:
            block_size = st.number_input("Context Size", value=selected_config.block_size, min_value=8, max_value=256)
        with adv_col2:
            n_embd = st.selectbox("Embedding Dim", [32, 48, 64, 96, 128, 192, 256], 
                                  index=list([32, 48, 64, 96, 128, 192, 256]).index(selected_config.n_embd))
        with adv_col3:
            n_head = st.selectbox("Attention Heads", [1, 2, 4, 8, 16],
                                  index=list([1, 2, 4, 8, 16]).index(selected_config.n_head if selected_config.n_head in [1, 2, 4, 8, 16] else 4))
        with adv_col4:
            n_layer = st.selectbox("Layers", [2, 3, 4, 6, 8, 12],
                                  index=list([2, 3, 4, 6, 8, 12]).index(selected_config.n_layer if selected_config.n_layer in [2, 3, 4, 6, 8, 12] else 4))
        
        st.markdown("**Training Optimizations (Phase 2)**")
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        
        with opt_col1:
            use_lr_scheduling = st.checkbox("LR Scheduling", value=True, help="Cosine annealing + warmup")
        with opt_col2:
            use_gradient_clipping = st.checkbox("Gradient Clipping", value=True, help="Norm=1.0")
        with opt_col3:
            use_early_stopping = st.checkbox("Early Stopping", value=False, help="Stop if val plateaus")
        with opt_col4:
            enable_checkpointing = st.checkbox("Checkpointing", value=False, help="Save every 5 epochs")
        
        st.markdown("**Device / Precision (Phase 3)**")
        device_col1, device_col2 = st.columns(2)
        available_devices = ["cpu"]
        if Config.detect_device() == "cuda":
            available_devices.append("cuda")
        with device_col1:
            training_device = st.selectbox("Training Device", available_devices, index=len(available_devices)-1)
        with device_col2:
            use_mixed_precision = st.checkbox("Mixed Precision", value=False, help="Use AMP when running on CUDA")
        
        st.markdown("**Regularization**")
        other_col1, other_col2 = st.columns(2)
        with other_col1:
            dropout = st.slider("Dropout", 0.0, 0.5, selected_config.dropout, 0.05)
        with other_col2:
            use_rmsnorm = st.checkbox("Use RMSNorm", value=False, help="Faster normalization")
    else:
        # Use preset values
        block_size = selected_config.block_size
        n_embd = selected_config.n_embd
        n_head = selected_config.n_head
        n_layer = selected_config.n_layer
        dropout = selected_config.dropout
        use_lr_scheduling = True
        use_gradient_clipping = True
        use_early_stopping = False
        enable_checkpointing = False
        use_rmsnorm = False
        training_device = "cpu"
        use_mixed_precision = False
    
    if st.button("Start Training"):
        try:
            if 'raw_text' not in st.session_state or not st.session_state.raw_text:
                st.error("⚠️ Please provide and parse data in the 'Data' tab first.")
                logger.warning("Training attempted without parsed data")
            else:
                with st.spinner("🔄 Training in progress..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(iter_num, max_iters, loss):
                        progress = min(iter_num / max_iters, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Iteration {iter_num}/{max_iters} - Loss: {loss:.4f}")

                    hyperparams = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'block_size': block_size,
                        'n_embd': n_embd,
                        'n_head': n_head,
                        'n_layer': n_layer,
                        'dropout': dropout,
                        'use_rmsnorm': use_rmsnorm,
                    }
                    
                    try:
                        result = train_model(
                            st.session_state.raw_text, 
                            hyperparams,
                            use_lr_scheduling=use_lr_scheduling,
                            use_gradient_clipping=use_gradient_clipping,
                            use_early_stopping=use_early_stopping,
                            enable_checkpointing=enable_checkpointing,
                            device=training_device,
                            use_mixed_precision=use_mixed_precision,
                            progress_callback=update_progress
                        )
                        
                        # Handle both old and new return formats
                        if len(result) == 6:
                            model, tokenizer, final_loss, train_loss_history, val_loss_history, accuracy_history = result
                        elif len(result) == 5:
                            model, tokenizer, final_loss, train_loss_history, val_loss_history = result
                            accuracy_history = []
                        else:
                            model, tokenizer, final_loss = result
                            train_loss_history = []
                            val_loss_history = []
                            accuracy_history = []
                        
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.training_loss_history = train_loss_history
                        st.session_state.val_loss_history = val_loss_history
                        st.session_state.accuracy_history = accuracy_history
                        st.session_state.training_stats = {
                            'final_loss': final_loss,
                            'hyperparams': hyperparams,
                            'training_loss_history': train_loss_history,
                            'val_loss_history': val_loss_history,
                            'accuracy_history': accuracy_history,
                        }
                        
                        st.success(f"✅ Training Complete! Final Loss: {final_loss:.4f}")
                        logger.info(f"Training completed successfully. Final loss: {final_loss:.4f}")
                        
                    except ValueError as e:
                        st.error(f"❌ Input Validation Error: {str(e)}")
                        logger.error(f"Input validation during training: {e}")
                    except RuntimeError as e:
                        st.error(f"❌ Training Error: {str(e)}")
                        logger.error(f"Runtime error during training: {e}", exc_info=True)
                    except Exception as e:
                        st.error(f"❌ Unexpected error during training: {str(e)}")
                        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Error in training section: {e}", exc_info=True)
    
    # Display training metrics graphs (only in advanced mode)
    if show_advanced and len(st.session_state.training_loss_history) > 0:
        st.markdown("---")
        st.subheader("📊 Training Metrics")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.markdown("**Training & Validation Loss**")
            fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
            epochs_range = range(1, len(st.session_state.training_loss_history) + 1)
            ax_loss.plot(epochs_range, st.session_state.training_loss_history, 'b-o', label='Training Loss', linewidth=2, markersize=4)
            if len(st.session_state.val_loss_history) > 0:
                ax_loss.plot(epochs_range, st.session_state.val_loss_history, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
            ax_loss.set_xlabel('Epoch', fontsize=11)
            ax_loss.set_ylabel('Loss', fontsize=11)
            ax_loss.set_title('Training Progress', fontsize=12, fontweight='bold')
            ax_loss.legend(loc='upper right')
            ax_loss.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_loss)
        
        with metrics_col2:
            st.markdown("**Validation Accuracy**")
            if hasattr(st.session_state, 'accuracy_history') and len(st.session_state.accuracy_history) > 0:
                fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
                epochs_range_acc = range(1, len(st.session_state.accuracy_history) + 1)
                ax_acc.plot(epochs_range_acc, st.session_state.accuracy_history, 'g-^', label='Validation Accuracy', linewidth=2, markersize=4)
                ax_acc.set_xlabel('Epoch', fontsize=11)
                ax_acc.set_ylabel('Accuracy (%)', fontsize=11)
                ax_acc.set_title('Accuracy Progress', fontsize=12, fontweight='bold')
                ax_acc.legend(loc='lower right')
                ax_acc.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_acc)
            else:
                st.info("Accuracy data not available for this training session.")
        
        with metrics_col3:
            st.markdown("**Training Summary**")
            if len(st.session_state.training_loss_history) > 0:
                st.metric(
                    "Initial Training Loss",
                    f"{st.session_state.training_loss_history[0]:.4f}"
                )
                st.metric(
                    "Final Training Loss",
                    f"{st.session_state.training_loss_history[-1]:.4f}"
                )
                if len(st.session_state.val_loss_history) > 0:
                    st.metric(
                        "Best Validation Loss",
                        f"{min(st.session_state.val_loss_history):.4f}"
                    )
                if hasattr(st.session_state, 'accuracy_history') and len(st.session_state.accuracy_history) > 0:
                    st.metric(
                        "Best Validation Accuracy",
                        f"{max(st.session_state.accuracy_history):.2f}%"
                    )

# --- TAB 3: INFERENCE ---
with tabs[2]:
    st.header("Test Your Model")
    
    model_source = st.radio(
        "Select model source",
        ("Trained Model", "Prebuilt Model"),
        index=0,
        help="Choose between your trained model or a prebuilt local model."
    )

    selected_prebuilt_model = None
    is_cedar_model = False
    if model_source == "Prebuilt Model":
        available_models = list(PREBUILT_MODEL_ALIASES.keys())
        selected_prebuilt_model = st.selectbox(
            "Prebuilt model",
            available_models,
            index=0,
            help="Select a prebuilt local transformer model. It will be downloaded if not present."
        )
        is_cedar_model = selected_prebuilt_model == "Cedar"
        if st.button("Load Prebuilt LLM"):
            with st.spinner("Loading prebuilt model... This may take a few seconds."):
                try:
                    # Resolve the model alias
                    from prebuilt import resolve_prebuilt_model_name
                    resolved_name = resolve_prebuilt_model_name(selected_prebuilt_model)
                    
                    # Check if model is downloaded
                    model_path = get_prebuilt_model_path(selected_prebuilt_model)
                    if not model_path or not os.path.exists(model_path):
                        st.info(f"📥 {selected_prebuilt_model} model not found locally. Downloading...")
                        if not download_prebuilt_model_if_needed(resolved_name):
                            raise RuntimeError(f"Failed to download {selected_prebuilt_model}")
                    
                    # Now load the model
                    st.session_state.prebuilt_llm = load_prebuilt_model(model_path)
                    st.session_state.prebuilt_model_path = model_path
                    st.session_state.prebuilt_loaded = True
                    st.success(f"✅ Loaded prebuilt model: {selected_prebuilt_model}")
                    logger.info(f"Loaded prebuilt model from {model_path}")
                except RuntimeError as e:
                    st.session_state.prebuilt_llm = None
                    st.session_state.prebuilt_model_path = None
                    st.session_state.prebuilt_loaded = False
                    st.error(f"❌ Could not load prebuilt model: {e}")
                    logger.warning(f"Prebuilt model load failed: {e}")
                except Exception as e:
                    st.session_state.prebuilt_llm = None
                    st.session_state.prebuilt_model_path = None
                    st.session_state.prebuilt_loaded = False
                    st.error(f"❌ Could not load prebuilt model: {e}")
                    logger.error(f"Prebuilt model load failed: {e}", exc_info=True)

    if model_source == "Trained Model" and st.session_state.model is None:
        st.warning("⚠️ Please train a model first.")
    elif model_source == "Prebuilt Model" and not st.session_state.prebuilt_loaded:
        st.warning("⚠️ Please load a prebuilt model above before generating.")
    else:
        try:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                temperature = st.slider("Temperature", 0.1, 2.0, 0.8, help="Controls randomness. Higher is more creative.")
            with col_p2:
                top_p = st.slider("Top-P (Nucleus Sampling)", 0.0, 1.0, 0.9, help="Limits the token pool to the top cumulative probability.")
                
            user_query = st.text_input("Enter your message:", placeholder="Hello")
            enable_web_access = st.checkbox("Enable web access", value=False, help="Allow the model to use live web scraping data.")
            web_input = ""
            if enable_web_access:
                web_input = st.text_input("Web URL or search query", placeholder="https://example.com or latest world news")

            advanced_inference = st.checkbox("Show advanced inference options", value=False)
            if advanced_inference:
                with st.expander("Advanced Inference Controls", expanded=True):
                    cedar_defaults = {
                        'max_new_tokens': 200,
                        'temperature': 0.4,
                        'top_p': 0.85,
                        'top_k': 50,
                        'repetition_penalty': 1.2,
                        'do_sample': True,
                        'eos_token_id': 0,
                        'pad_token_id': 0,
                        'truncation': True,
                    }
                    default_values = cedar_defaults if is_cedar_model else {
                        'max_new_tokens': 100,
                        'temperature': 0.8,
                        'top_p': 0.95,
                        'top_k': 50,
                        'repetition_penalty': 1.0,
                        'do_sample': True,
                        'eos_token_id': 0,
                        'pad_token_id': 0,
                        'truncation': True,
                    }

                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        max_new_tokens = st.number_input(
                            "Max New Tokens",
                            min_value=1,
                            max_value=1024,
                            value=default_values['max_new_tokens'],
                            step=1
                        )
                        top_k = st.number_input(
                            "Top-K",
                            min_value=0,
                            max_value=1024,
                            value=default_values['top_k'],
                            step=1,
                            help="Restrict generation to the top K probable tokens."
                        )
                        repetition_penalty = st.number_input(
                            "Repetition Penalty",
                            min_value=0.1,
                            max_value=3.0,
                            value=default_values['repetition_penalty'],
                            step=0.1
                        )
                        eos_token_id = st.number_input(
                            "EOS Token ID",
                            min_value=0,
                            value=default_values['eos_token_id'],
                            step=1,
                            help="Token ID to use for end-of-sequence detection."
                        )
                    with col_a2:
                        do_sample = st.checkbox("Do Sample", value=default_values['do_sample'])
                        pad_equals_eos = st.checkbox(
                            "Pad Token ID = EOS Token ID",
                            value=True,
                            help="Use the EOS token ID as the padding token ID for consistency."
                        )
                        if pad_equals_eos:
                            pad_token_id = eos_token_id
                            st.caption(f"Pad Token ID is set to EOS Token ID ({eos_token_id}).")
                        else:
                            pad_token_id = st.number_input(
                                "Pad Token ID",
                                min_value=0,
                                value=default_values['pad_token_id'],
                                step=1,
                            )
                        truncation = st.checkbox(
                            "Enable Truncation",
                            value=default_values['truncation'],
                            help="Truncate long prompts to fit the model context length."
                        )
            else:
                max_new_tokens = 100
                top_k = 50
                repetition_penalty = 1.0
                do_sample = True
                eos_token_id = 0
                pad_token_id = 0
                truncation = True

            if st.button("Generate"):
                try:
                    if not user_query or len(user_query.strip()) == 0:
                        st.error("❌ Please enter a message.")
                        logger.warning("Generation attempted without input")
                    else:
                        current_datetime = get_current_datetime()
                        is_time_query = is_datetime_question(user_query)
                        if is_time_query:
                            response = f"The current date and time is {current_datetime}."
                            st.success("✅ Date/time query answered with live system clock.")
                            st.text_area("AI Response:", value=response.strip(), height=150, disabled=True)
                            logger.info(f"Answered date/time query: {user_query[:50]}...")
                        else:
                            web_context = ""
                            if enable_web_access and web_input.strip():
                                web_context = fetch_web_data(web_input.strip())
                                if web_context:
                                    st.info("🌐 Web context loaded successfully.")
                                else:
                                    st.warning("⚠️ Web context could not be loaded. Proceeding without web data.")

                            prompt_prefix = f"CURRENT_DATETIME: {current_datetime}\n"
                            if web_context:
                                prompt_prefix += f"WEB_CONTEXT: {web_context}\n"

                            if model_source == "Prebuilt Model" and is_cedar_model:
                                system_prompt = (
                                    "You are Cedar, a helpful assistant. Keep your responses short and direct. Speak only in plain English. Never provide Python code or programming explanations. If math is involved, show the calculation as (a + b = c). Be direct and brief."
 


                                )
                                prompt = (
                                    f"{system_prompt}\n"
                                    f"<|im_start|>user\n{user_query}<|im_end|>\n"
                                    f"<|im_start|>assistant\n"
                                )
                            else:
                                prompt = f"{prompt_prefix}USER INPUT: {user_query}\nOUTPUT:"

                            if model_source == "Trained Model":
                                context_idxs = st.session_state.tokenizer.encode(prompt)
                                if len(context_idxs) == 0:
                                    st.error("❌ Could not encode input message")
                                    logger.error("Encoding failed for input message")
                                else:
                                    if truncation and len(context_idxs) > st.session_state.model.block_size:
                                        context_idxs = context_idxs[-st.session_state.model.block_size:]

                                    context_tensor = torch.tensor(context_idxs, dtype=torch.long).unsqueeze(0)
                                    device = Config.detect_device()
                                    st.session_state.model.to(device)
                                    context_tensor = context_tensor.to(device)
                                    
                                    with st.spinner("🤔 Thinking..."):
                                        try:
                                            generated_idxs = st.session_state.model.generate(
                                                context_tensor,
                                                max_new_tokens=max_new_tokens,
                                                temperature=temperature,
                                                top_p=top_p,
                                                top_k=top_k,
                                                repetition_penalty=repetition_penalty,
                                                do_sample=do_sample,
                                                eos_token_id=eos_token_id,
                                            )
                                            decoded_output = st.session_state.tokenizer.decode(generated_idxs[0].tolist())
                                            response = decoded_output[len(prompt):]
                                            if "USER INPUT:" in response:
                                                response = response.split("USER INPUT:")[0]
                                            
                                            st.success("✅ Generation complete!")
                                            st.text_area("AI Response:", value=response.strip(), height=150, disabled=True)
                                            st.caption(f"**Full generation length:** {len(decoded_output)} characters")
                                            logger.info(f"Successfully generated response for query: {user_query[:50]}...")
                                        except Exception as gen_error:
                                            st.error(f"❌ Error during generation: {str(gen_error)}")
                                            logger.error(f"Generation error: {gen_error}", exc_info=True)
                            else:
                                with st.spinner("Thinking..."):
                                    try:
                                        generated_text = generate_with_prebuilt_model(
                                            st.session_state.prebuilt_llm,
                                            prompt,
                                            max_tokens=max_new_tokens,
                                            temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k,
                                            repetition_penalty=repetition_penalty,
                                            do_sample=do_sample,
                                            eos_token_id=eos_token_id,
                                            pad_token_id=pad_token_id,
                                            truncation=truncation,
                                            stop=["<|im_start|>user", "USER INPUT:"]
                                        )
                                        response = generated_text.strip()
                                        st.success("✅ Generation complete with prebuilt model!")
                                        st.text_area("AI Response:", value=response, height=150, disabled=True)
                                        st.caption(f"**Full generation length:** {len(response)} characters")
                                        logger.info(f"Successfully generated response for query: {user_query[:50]} using prebuilt LLM")
                                    except Exception as gen_error:
                                        st.error(f"❌ Error during prebuilt model generation: {str(gen_error)}")
                                        logger.error(f"Prebuilt generation error: {gen_error}", exc_info=True)
                except Exception as e:
                    st.error(f"❌ Error processing input: {str(e)}")
                    logger.error(f"Error in inference: {e}", exc_info=True)
        except Exception as e:
            st.error(f"❌ Error setting up inference: {str(e)}")
            logger.error(f"Error setting up inference: {e}", exc_info=True)

# --- TAB 4: MODEL INFO ---
with tabs[3]:
    st.header("Model Parameters & Stats")
    
    if st.session_state.model is None:
        st.info("ℹ️ No model trained yet. Train a model to see its parameters.")
    else:
        try:
            stats = st.session_state.training_stats
            hp = stats.get('hyperparams', {})
            final_loss_str = f"{stats.get('final_loss', 'N/A'):.4f}" if isinstance(stats.get('final_loss'), (int, float)) else str(stats.get('final_loss', 'N/A'))
            
            st.subheader("📊 Training Hyperparameters")
            st.markdown(f"""
            <div style="display: flex; flex-wrap: wrap;">
                <div class="model-tag">Learning Rate: {hp.get('learning_rate')}</div>
                <div class="model-tag">Batch Size: {hp.get('batch_size')}</div>
                <div class="model-tag">Epochs: {hp.get('epochs')}</div>
                <div class="model-tag">Final Loss: {final_loss_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🏗 Architecture Specs")
            # Count parameters
            n_params = sum(p.numel() for p in st.session_state.model.parameters())
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                - **Total Parameters:** `{n_params:,}`
                - **Vocab Size:** `{st.session_state.tokenizer.vocab_size}`
                """)
            with col_b:
                st.markdown(f"""
                - **Embedding Dim:** `64`
                - **Attention Heads:** `4`
                - **Transformer Layers:** `4`
                """)

            if st.session_state.prebuilt_loaded and st.session_state.prebuilt_model_path:
                st.subheader("🧠 Prebuilt Model Loaded")
                st.markdown(f"- **Model Path:** `{st.session_state.prebuilt_model_path}`")
            
            st.markdown("---")
            st.subheader("💾 Save Trained Model")
            save_model_name = st.text_input("Model name", value="", key="save_model_name_input", help="Give your trained model a friendly name for later reuse.")
            if st.button("Save Current Model", key="save_current_model"):
                if st.session_state.model is None or st.session_state.tokenizer is None:
                    st.error("❌ No trained model available to save.")
                elif not save_model_name.strip():
                    st.error("❌ Please enter a name for the saved model.")
                else:
                    try:
                        metadata = {
                            'model_config': st.session_state.model.get_config(),
                            'training_stats': st.session_state.training_stats,
                        }
                        save_dir = save_trained_model(save_model_name, st.session_state.model, st.session_state.tokenizer, metadata)
                        st.success(f"✅ Model saved to: {save_dir}")
                        logger.info(f"Saved trained model to {save_dir}")
                    except Exception as e:
                        st.error(f"❌ Failed to save model: {e}")
                        logger.error(f"Save trained model failed: {e}", exc_info=True)
        except Exception as e:
            st.error(f"❌ Error displaying model info: {str(e)}")
            logger.error(f"Error in Model Info tab: {e}", exc_info=True)
