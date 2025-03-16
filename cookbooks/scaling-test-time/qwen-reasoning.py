import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
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
eval_dataset = tokenized_dataset["test"]


lora_config = LoraConfig(
    r=16,              # Rank of the low-rank matrices
    lora_alpha=32,     # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05, # Dropout for regularization
    bias="none",       # No bias adaptation
    task_type="CAUSAL_LM"  
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="./qwen2.5-gsm8k-lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Simulate larger batch size
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # Mixed precision training
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    warmup_steps=100,
    remove_unused_columns=False,
    report_to="wandb"  # Disable wandb or other logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()