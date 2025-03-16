import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device : {device}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16,
    device_map = "auto",
    load_in_4bit = True
)
model.to(device)

dataset = load_dataset("gsm8k", "main")

