import os
from typing import List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None


class PrebuiltModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        truncation: bool = True,
        stop: Optional[List[str]] = None,
    ) -> str:
        tokenizer_args = {
            "return_tensors": "pt",
            "truncation": truncation,
        }
        max_length = getattr(self.tokenizer, "model_max_length", None)
        if max_length is not None and truncation:
            tokenizer_args["max_length"] = max_length

        inputs = self.tokenizer(prompt, **tokenizer_args)
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
            }
            if top_k is not None and top_k > 0:
                generation_kwargs["top_k"] = top_k
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id

            outputs = self.model.generate(**inputs, **generation_kwargs)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        prompt_clean = prompt.strip()
        prompt_no_tokens = prompt_clean.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

        if prompt_clean and generated_text.startswith(prompt_clean):
            generated_text = generated_text[len(prompt_clean):].strip()
        elif prompt_no_tokens and generated_text.startswith(prompt_no_tokens):
            generated_text = generated_text[len(prompt_no_tokens):].strip()
        elif prompt_no_tokens and prompt_no_tokens in generated_text:
            generated_text = generated_text.split(prompt_no_tokens, 1)[1].strip()

        for prefix in (
            "<|im_start|>assistant",
            "<|im_start|>user",
            "<|im_end|>",
            "assistant\n",
            "assistant:",
            "user\n",
            "user:",
        ):
            if generated_text.startswith(prefix):
                generated_text = generated_text[len(prefix):].strip()

        if stop:
            for token in stop:
                if token in generated_text:
                    generated_text = generated_text.split(token, 1)[0].strip()
        return generated_text


PREBUILT_MODEL_ALIASES = {
    "Talkative Dumbo": "smollm2",
    "Cedar": "qwen",
}


def get_prebuilt_model_dir() -> str:
    """Return the local directory containing prebuilt models."""
    return os.path.join(os.path.dirname(__file__), "prebuilt", "models")


def resolve_prebuilt_model_name(model_name: str) -> str:
    """Translate a display name to the actual local model folder name."""
    if model_name in PREBUILT_MODEL_ALIASES:
        return PREBUILT_MODEL_ALIASES[model_name]
    return model_name


def list_prebuilt_models() -> List[str]:
    """Return a list of available prebuilt model entries."""
    model_dir = get_prebuilt_model_dir()
    if not os.path.isdir(model_dir):
        return []

    models = []
    for entry in os.listdir(model_dir):
        path = os.path.join(model_dir, entry)
        if os.path.isdir(path) or os.path.isfile(path):
            alias = next((name for name, real in PREBUILT_MODEL_ALIASES.items() if real == entry), None)
            models.append(alias or entry)
    return sorted(models)


def get_prebuilt_model_path(model_name: Optional[str] = None) -> Optional[str]:
    """Return the full path to a requested prebuilt model."""
    model_dir = get_prebuilt_model_dir()
    if not os.path.isdir(model_dir):
        return None

    if model_name:
        resolved_name = resolve_prebuilt_model_name(model_name)
        candidate = os.path.join(model_dir, resolved_name)
        if os.path.isdir(candidate) or os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Prebuilt model not found: {model_name}")

    models = list_prebuilt_models()
    return os.path.join(model_dir, resolve_prebuilt_model_name(models[0])) if models else None


def load_prebuilt_model(model_path: Optional[str] = None):
    """Load a local prebuilt Hugging Face transformer model."""
    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise ImportError(
            "transformers and torch are required to load prebuilt models. Install them with `pip install -r requirements.txt`."
        )

    if model_path is None:
        model_path = get_prebuilt_model_path()

    if model_path is None or not (os.path.isdir(model_path) or os.path.isfile(model_path)):
        raise FileNotFoundError("No prebuilt model found in prebuilt/models.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        return PrebuiltModel(model, tokenizer)
    except Exception as e:
        raise RuntimeError(
            "Failed to load the prebuilt model. Ensure the selected item in prebuilt/models is a valid local Hugging Face transformer model folder."
        ) from e


def generate_with_prebuilt_model(
    llm,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    truncation: bool = True,
    stop: Optional[List[str]] = None,
) -> str:
    """Generate text from a prebuilt LLM instance."""
    if llm is None:
        raise ValueError("Prebuilt model instance is not loaded.")

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        truncation=truncation,
        stop=stop,
    )
    return str(output).strip()

