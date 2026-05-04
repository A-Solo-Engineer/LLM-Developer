import streamlit as st
import torch
import os
import time
from utils import parse_training_data, prepare_corpus, SimpleTokenizer
from train import train_model
from model import GPTLanguageModel

st.set_page_config(
    page_title="LLM Developer Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1c24, #0e1117);
    }
    
    /* Custom Card Style */
    .css-1r6slb0, .stTabs {
        background-color: rgba(26, 28, 36, 0.6);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d1ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d1ff, #007bff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 209, 255, 0.4);
    }
    
    /* Input areas */
    .stTextArea textarea, .stTextInput input {
        background-color: #1a1c24 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #fafafa !important;
    }
    
    /* Tags for Model Info */
    .model-tag {
        display: inline-block;
        background: rgba(0, 209, 255, 0.1);
        color: #00d1ff;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(0, 209, 255, 0.3);
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Success/Info boxes */
    .stAlert {
        background-color: rgba(26, 28, 36, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with logo and status
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("LLM Studio")
    st.markdown("---")
    if 'model' in st.session_state and st.session_state.model:
        st.success("🟢 Model Ready")
    else:
        st.info("🔴 No Model Trained")
    
    st.markdown("---")
    st.caption("v1.0.0 | Built by Aarav")

st.title("🚀 LLM Developer Studio")
st.markdown("### Crafting Custom Intelligence on your CPU")

# Session State for Model and Tokenizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'training_stats' not in st.session_state:
    st.session_state.training_stats = {}

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
        parsed_data = parse_training_data(data_input)
        st.write(f"Found {len(parsed_data)} training pairs.")
        st.json(parsed_data[:5])
        st.session_state.parsed_data = parsed_data
        st.session_state.raw_text = prepare_corpus(parsed_data)

# --- TAB 2: TRAINING ---
with tabs[1]:
    st.header("Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        lr = st.number_input("Learning Rate", value=0.001, format="%.4f")
    with col2:
        batch_size = st.number_input("Batch Size", value=32, min_value=1, max_value=128)
    with col3:
        epochs = st.number_input("Epochs", value=50, min_value=1, max_value=1000)
        
    if st.button("Start Training"):
        if 'raw_text' not in st.session_state or not st.session_state.raw_text:
            st.error("Please provide and parse data in the 'Data' tab first.")
        else:
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
                'block_size': 64 # Fixed for simplicity/speed on CPU
            }
            
            with st.spinner("Training in progress..."):
                model, tokenizer, final_loss = train_model(
                    st.session_state.raw_text, 
                    hyperparams, 
                    progress_callback=update_progress
                )
                
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.training_stats = {
                'final_loss': final_loss,
                'hyperparams': hyperparams
            }
            
            st.success(f"Training Complete! Final Loss: {final_loss:.4f}")

# --- TAB 3: INFERENCE ---
with tabs[2]:
    st.header("Test Your Model")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
    else:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, help="Controls randomness. Higher is more creative.")
        with col_p2:
            top_p = st.slider("Top-P (Nucleus Sampling)", 0.0, 1.0, 0.9, help="Limits the token pool to the top cumulative probability.")
            
        user_query = st.text_input("Enter your message:", placeholder="Hello")
        
        if st.button("Generate"):
            if not user_query:
                st.error("Please enter a message.")
            else:
                prompt = f"USER INPUT: {user_query}\nOUTPUT:"
                
                # Encode
                context_idxs = st.session_state.tokenizer.encode(prompt)
                context_tensor = torch.tensor(context_idxs, dtype=torch.long).unsqueeze(0) # (1, T)
                
                # Generate
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                st.session_state.model.to(device)
                context_tensor = context_tensor.to(device)
                
                with st.spinner("Thinking..."):
                    # Limit generation to avoid infinite loops if model is dumb
                    generated_idxs = st.session_state.model.generate(
                        context_tensor, 
                        max_new_tokens=100, 
                        temperature=temperature, 
                        top_p=top_p
                    )
                    
                decoded_output = st.session_state.tokenizer.decode(generated_idxs[0].tolist())
                
                # Parse out just the output part
                # The model generates the whole sequence including the prompt
                response = decoded_output[len(prompt):]
                # Stop at next USER INPUT if it generates one
                if "USER INPUT:" in response:
                    response = response.split("USER INPUT:")[0]
                
                st.text_area("AI Response:", value=response.strip(), height=150)
                st.caption(f"Full generation: {decoded_output}")

# --- TAB 4: MODEL INFO ---
with tabs[3]:
    st.header("Model Parameters & Stats")
    
    if st.session_state.model is None:
        st.write("No model trained yet.")
    else:
        stats = st.session_state.training_stats
        hp = stats.get('hyperparams', {})
        
        st.subheader("📊 Training Hyperparameters")
        st.markdown(f"""
        <div style="display: flex; flex-wrap: wrap;">
            <div class="model-tag">Learning Rate: {hp.get('learning_rate')}</div>
            <div class="model-tag">Batch Size: {hp.get('batch_size')}</div>
            <div class="model-tag">Epochs: {hp.get('epochs')}</div>
            <div class="model-tag">Final Loss: {stats.get('final_loss', 'N/A'):.4f}</div>
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
