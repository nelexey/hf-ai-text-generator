import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import shutil

# Login to Hugging Face
login(token="HF_TOKEN")


# Function to clear memory
def clear_memory():
    """Releases unused memory, especially for GPU if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create a temporary directory for model caching
os.makedirs('model_cache', exist_ok=True)

# Clear memory before loading the model
clear_memory()

# CUDA configuration
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir='model_cache'
)

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision
    low_cpu_mem_usage=True,  # Optimize CPU memory usage
    device_map="auto",  # Auto device allocation (GPU, CPU)
    cache_dir='model_cache',  # Cache directory for model
    max_memory={0: "11GB", "cpu": "16GB"},  # Memory limits for GPU and CPU
)

# Create generation pipeline
print("Creating pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Sample prompt for model response
context = """
USER: Can you give me an overview of the history and significance of the Silk Road?
YOU:
"""

try:
    # Clear memory before generating
    clear_memory()

    print("Generating response...")
    # Generate response with optimized parameters
    with torch.no_grad():
        response = pipe(
            context,
            max_length=150,  # Reduced generation length
            num_return_sequences=1,
            do_sample=True,
            batch_size=1,
            use_cache=True,
            temperature=0.7  # Temperature for stable generation
        )
    print(response[0]['generated_text'])

except Exception as e:
    print(f"Error during generation: {e}")

finally:
    # Clear memory after generation
    print("Clearing memory...")
    del model
    del tokenizer
    del pipe
    clear_memory()

    # Remove model cache directory
    try:
        shutil.rmtree('model_cache')
    except Exception as e:
        print(f"Error clearing cache: {e}")
