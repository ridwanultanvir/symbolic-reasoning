import torch
import re
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ───────────────────────────────────
# Load model and tokenizer
# ───────────────────────────────────
MODEL_DIR = "phi2_malls_finetuned"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

# ───────────────────────────────────
# Load test set and prepare prompts
# ───────────────────────────────────
ds = load_dataset("yuan-yang/MALLS-v0")
test_set = ds["test"]

def build_prompt(example):
    return f"### NL:\n{example['NL']}\n### FOL:\n"

prompts = [build_prompt(ex) for ex in test_set]

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
# Validation functions
# ───────────────────────────────────
def has_balanced_parentheses(s):
    return s.count('(') == s.count(')')

def has_valid_predicates(s):
    return bool(re.search(r'\b\w+\s*\(\s*\w+\s*\)', s))

# ───────────────────────────────────
# Categorize outputs
# ───────────────────────────────────
valid = 0
paren_error = 0
predicate_error = 0

for expr in predictions:
    paren_ok = has_balanced_parentheses(expr)
    pred_ok = has_valid_predicates(expr)

    if paren_ok and pred_ok:
        valid += 1
    elif not paren_ok:
        paren_error += 1
    elif not pred_ok:
        predicate_error += 1

# ───────────────────────────────────
# Pie chart
# ───────────────────────────────────
labels = ['Valid', 'Unbalanced Parentheses', 'Invalid Predicate Format']
sizes = [valid, paren_error, predicate_error]
colors = ['#4F81BD', '#D9534F', '#F0AD4E']
explode = [0.05, 0.07, 0.07]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, shadow=True)
plt.title('Syntactic Validity Breakdown of FOL Outputs')
plt.tight_layout()
plt.savefig("figures/5_syntactic_validity_pie.png")
plt.show()
