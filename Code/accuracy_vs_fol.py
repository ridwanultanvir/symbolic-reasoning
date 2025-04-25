import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import matplotlib.pyplot as plt

# ───────────────────────────────────
# Load model and tokenizer
# ───────────────────────────────────
MODEL_DIR = "phi2_malls_finetuned"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

# ───────────────────────────────────
# Load dataset and prepare prompts
# ───────────────────────────────────
ds = load_dataset("yuan-yang/MALLS-v0")
test_set = ds["test"]

def build_prompt(example):
    return f"### NL:\n{example['NL']}\n### FOL:\n"

prompts = [build_prompt(ex) for ex in test_set]
references = [ex["FOL"] for ex in test_set]

# ───────────────────────────────────
# Generate predictions
# ───────────────────────────────────
model.eval()
predictions = []

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = decoded.split("### FOL:\n")[-1].strip()
    predictions.append(prediction)

# ───────────────────────────────────
# Group by FOL length and compute accuracy
# ───────────────────────────────────
length_bins = [(5,9), (10,14), (15,19), (20,24), (25,29), (30,100)]
bin_labels = ['5–9', '10–14', '15–19', '20–24', '25–29', '30+']
bin_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

for ref, pred in zip(references, predictions):
    length = len(ref.split())
    matched = (ref.strip() == pred.strip())
    for (low, high), label in zip(length_bins, bin_labels):
        if low <= length <= high:
            bin_stats[label]['total'] += 1
            bin_stats[label]['correct'] += int(matched)
            break

# ───────────────────────────────────
# Compute accuracy per bin
# ───────────────────────────────────
x_labels = []
y_accuracy = []

for label in bin_labels:
    correct = bin_stats[label]['correct']
    total = bin_stats[label]['total']
    acc = correct / total if total > 0 else 0
    x_labels.append(label)
    y_accuracy.append(acc)

# ───────────────────────────────────
# Plot
# ───────────────────────────────────
plt.figure(figsize=(7, 5))
plt.plot(x_labels, y_accuracy, marker='o', linewidth=2, color='#4F81BD')
plt.ylim(0, 1)
plt.title('Accuracy vs. FOL Length')
plt.xlabel('FOL Length (Token Count Range)')
plt.ylabel('Exact Match Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)

for i, acc in enumerate(y_accuracy):
    plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("figures/3_accuracy_vs_length.png")
plt.show()
