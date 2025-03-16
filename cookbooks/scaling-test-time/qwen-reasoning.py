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
    device_map = "auto"
)
model.to(device)

dataset = load_dataset("gsm8k", "main")

# print(dataset['train'][0])

def preprocess_function(example):

    """
    a function to set the template of prompt according to the 'gsm8k' dataset
    """

    question = example["question"]
    answer = example["answer"]

    prompt = (
        f"Question: {question}\n"
        f"Reasoning: Let's solve this step-by-step, thinking deeply:\n"
        f"Answer : {answer}"
    )

    inputs = tokenizer(prompt, truncation = True, max_length = 512, padding = 'max_length')
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched = False, remove_columns = dataset["train"].column_names)

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["eval"]