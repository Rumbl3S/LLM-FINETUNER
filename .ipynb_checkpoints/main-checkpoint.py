import subprocess
import sys
import torch
import requests
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
from llm_finetuner.utils.ssh_tunnel import start_tunnel, stop_tunnel

# Remove the PyTorch MPS high watermark ratio environment variable if it exists
if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
    del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']

# Set the PyTorch MPS fallback environment variable
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Verify the environment variable is set
print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}")
print(f"PYTORCH_ENABLE_MPS_FALLBACK: {os.getenv('PYTORCH_ENABLE_MPS_FALLBACK')}")

# Function to install packages
def install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

# Install required packages
install("torch", "--pre", "--extra-index-url", "https://download.pytorch.org/whl/nightly/cpu")
install("transformers")
install("datasets")
install("accelerate")
# Optionally install bitsandbytes if supported
# install("bitsandbytes")

# Load environment variables
IMAGINATION_IP = os.getenv("IMAGINATION_IP")
SSH_USERNAME = os.getenv("SSH_USERNAME")
SSH_PKEY = os.getenv("SSH_PKEY")
IMAGINATION_PORT = os.getenv("IMAGINATION_PORT", "11434")  # Default to 11434 if not set
LOCAL_PORT = os.getenv("LOCAL_PORT")

# Print the environment variables to debug
print(f"IMAGINATION_IP: {IMAGINATION_IP}")
print(f"SSH_USERNAME: {SSH_USERNAME}")
print(f"SSH_PKEY: {SSH_PKEY}")
print(f"IMAGINATION_PORT: {IMAGINATION_PORT}")
print(f"LOCAL_PORT: {LOCAL_PORT}")

# Check for missing environment variables
if None in [IMAGINATION_IP, SSH_USERNAME, SSH_PKEY, LOCAL_PORT]:
    print("One or more environment variables are not set. Exiting.")
    exit(1)

# Convert port variables to integers
try:
    IMAGINATION_PORT = int(IMAGINATION_PORT)
    LOCAL_PORT = int(LOCAL_PORT)
except ValueError as e:
    print(f"Error converting ports to integers: {e}")
    exit(1)

def get_response(prompt, past_responses=None):
    url = f"http://0.0.0.0:{LOCAL_PORT}/api/chat"
    if past_responses is None:
        history = []
    else:
        history = [
            {"role": "assistant", "content": message} for message in past_responses
        ]
    history.append({"role": "user", "content": prompt})
    data = {
        "model": "llama3:70b",
        "messages": history,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        print(response.json())
        print(f"Request failed with status code {response.status_code}")
        return None

def train_model():
    if not torch.backends.mps.is_available():
        print("MPS device not found. Exiting.")
        exit(1)

    model_id = "gpt2"  # Switch to a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    model.to("mps")
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Print all module names to identify the correct target modules
    for name, module in model.named_modules():
        print(name)
    
    # Use the identified target modules for GPT-2
    target_modules = ["attn.c_attn", "attn.c_proj"]

    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=target_modules, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    
    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print_trainable_parameters(model)

    # Load the dataset
    try:
        data = load_dataset("Abirate/english_quotes")
        data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,  # Reduce batch size
            gradient_accumulation_steps=2,  # Reduce gradient accumulation steps
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir="outputs",
            save_steps=5,  # Save checkpoints less frequently
            save_total_limit=1,  # Only keep the last checkpoint
            bf16=True  # Use bf16 mixed precision for MPS compatibility
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!
    torch.cuda.empty_cache()  # Clear cache to free up memory before training
    trainer.train()
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained("outputs")

def run_inference(text):
    if not torch.backends.mps.is_available():
        print("MPS device not found. Exiting.")
        exit(1)

    model_id = "gpt2"  # Switch to a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to("mps")

    lora_config = LoraConfig.from_pretrained('outputs')
    model = get_peft_model(model, lora_config)

    device = torch.device("mps")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Move necessary parts to CPU for specific operation
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    model = model.to("cpu")
    
    outputs = model.generate(**inputs, max_new_tokens=20)
    
    # Move model back to MPS if needed
    model.to(device)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("Starting SSH tunnel...")
    start_tunnel(
        remote_server=IMAGINATION_IP,
        ssh_username=SSH_USERNAME,
        ssh_pkey=SSH_PKEY,
        remote_port=IMAGINATION_PORT,
        local_port=LOCAL_PORT,
    )

    try:
        print("SSH tunnel started. Training model...")
        train_model()
        
        print("Model training complete. Running inference...")
        inference_text = "Elon Musk "
        result = run_inference(inference_text)
        print(f"Inference result: {result}")
    finally:
        print("Stopping SSH tunnel...")
        stop_tunnel()
        print("SSH tunnel stopped.")
