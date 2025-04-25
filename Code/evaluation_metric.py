import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ───────────────────────────────────
# Load fine-tuned model and tokenizer
# ───────────────────────────────────
MODEL_DIR = "phi2_malls_finetuned"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

# ───────────────────────────────────
# Load and preprocess MALLS test set
# ───────────────────────────────────
ds = load_dataset("yuan-yang/MALLS-v0")
test_set = ds["test"]

def build_prompt(example):
    return f"### NL:\n{example['NL']}\n### FOL:\n"

prompts = [build_prompt(ex) for ex in test_set]
references = [ex["FOL"] for ex in test_set]

# ───────────────────────────────────
# Generate model predictions
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
    # Extract only the generated FOL part
    prediction = decoded.split("### FOL:\n")[-1].strip()
    predictions.append(prediction)

# ───────────────────────────────────
# Evaluation: Exact match + BLEU
# ───────────────────────────────────
exact_matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
exact_accuracy = exact_matches / len(references) * 100

bleu_scores = [
    sentence_bleu([ref.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
    for ref, pred in zip(references, predictions)
]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

# ───────────────────────────────────
# Token-level Precision, Recall, F1
# ───────────────────────────────────
y_true_flat = []
y_pred_flat = []

for ref, pred in zip(references, predictions):
    ref_tokens = ref.split()
    pred_tokens = pred.split()
    max_len = max(len(ref_tokens), len(pred_tokens))
    ref_tokens += ["PAD"] * (max_len - len(ref_tokens))
    pred_tokens += ["PAD"] * (max_len - len(pred_tokens))
    y_true_flat.extend(ref_tokens)
    y_pred_flat.extend(pred_tokens)

y_true_bin = [1 if t1 == t2 else 0 for t1, t2 in zip(y_true_flat, y_pred_flat)]
y_pred_bin = [1] * len(y_true_bin)  # model always predicts something

precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

# ───────────────────────────────────
# Print Results
# ───────────────────────────────────
print(f"Exact Match Accuracy: {exact_accuracy:.4f}%")
print(f"Average BLEU Score:   {avg_bleu:.4f}")
print(f"Token-Level Precision: {precision:.4f}")
print(f"Token-Level Recall:    {recall:.4f}")
print(f"Token-Level F1 Score:  {f1:.4f}")
