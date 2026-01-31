import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

def setup_experiment(method="DoRA", model_id="meta-llama/Meta-Llama-3-8B"):
    print(f"Setting up experiment for: {method}")
    
    # 1. Quantization Config for QLoRA
    bnb_config = None
    if method == "QLoRA":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # 2. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if method != "QLoRA" else None
    )

    # 3. Prepare for k-bit training (only for QLoRA)
    if method == "QLoRA":
        model = prepare_model_for_kbit_training(model)

    # 4. PEFT Configuration
    # use_dora=True enables the Weight Decomposition logic
    use_dora = True if method == "DoRA" else False
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora 
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

# Mock Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit", # Crucial for QLoRA memory saving
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    max_steps=1000, # Demo steps
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

