import sys, os, traceback, time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import login

# ───────────────────────────────────
# Dual Logger Setup
# ───────────────────────────────────
LOG_FILE = "train_phi2.log"
class Tee:
    def __init__(self, filename):
        self.file = open(filename, "a", buffering=1, encoding="utf-8")
        self.tty = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.tty.write(data)
    def flush(self):
        self.file.flush()
        self.tty.flush()
sys.stdout = sys.stderr = Tee(LOG_FILE)
print(f"\n=== Training started at {time.asctime()} ===\n")

# ───────────────────────────────────
# Config
# ───────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "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")
MODEL_ID = "microsoft/phi-2"
OUT_DIR = "phi2_malls_finetuned"
MAX_LEN = 512
EPOCHS = 5

# ───────────────────────────────────
# Main Function
# ───────────────────────────────────
def main():
    login(HF_TOKEN)
    print("Hugging Face authentication complete.")

    # Load dataset
    ds = load_dataset("yuan-yang/MALLS-v0")
    print("MALLS dataset loaded.")

    # Format: input is NL, label is FOL
    def format_pair(example):
        return {
            "prompt": f"### NL:\n{example['NL']}\n### FOL:\n",
            "target": example["FOL"]
        }

    ds = ds.map(format_pair)
    ds = ds.remove_columns(["NL", "FOL"])

    # Tokenizer and model
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.gradient_checkpointing_enable()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded and placed on device.")

    # Tokenize full prompt + target (label = same as input_ids)
    def tokenize(example):
        full_input = example["prompt"] + example["target"]
        enc = tok(full_input, truncation=True, padding="max_length", max_length=MAX_LEN)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["prompt", "target"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Training arguments
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_total_limit=2,
        logging_steps=50,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        tokenizer=tok,
        data_collator=collator,
    )

    print("Starting training...\n")
    trainer.train()
    print("\nTraining complete.")

    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"Model saved to: {OUT_DIR}")

if __name__ == "__main__":
    try:
        main()
        print(f"\n=== Training completed at {time.asctime()} ===\n")
        sys.exit(0)
    except Exception as e:
        print("\nError during training:\n")
        traceback.print_exc()
        print(f"\n=== Training FAILED at {time.asctime()} ===\n")
        sys.exit(1)
