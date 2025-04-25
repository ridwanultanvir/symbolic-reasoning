# Symbolic-Reasoning

**Toward Valid Symbolic Representations: Translating Natural Language to First-Order Logic**

This repository contains all code, figures, and analysis associated with a project that explores the translation of natural language (NL) into first-order logic (FOL) using a fine-tuned version of the `phi-2` language model. The experiments are conducted on the [MALLS-v0 dataset](https://huggingface.co/datasets/yuan-yang/MALLS-v0), and the project emphasizes syntactic validity, logical complexity, and performance under varying resource constraints.

---

## 🧠 Project Objective

Develop a lightweight symbolic translation system that maps natural language into formally valid logical expressions using a small language model under limited compute.

---

## 📁 Repository Structure

    Symbolic-Reasoning/
    ├── Code/
    │   ├── train_phi2.py              # Training script using Hugging Face Trainer
    │   ├── evaluation_metric.py       # Calculates BLEU, F1, precision, recall
    │   ├── accuracy_vs_fol.py         # Accuracy vs logical formula length
    │   ├── syntactic_validity.py      # Syntactic validation of generated FOL
    │   ├── dataset_analysis.ipynb     # Exploratory analysis of MALLS dataset
    │   └── train_phi2.log             # Training log file
    │
    ├── Report/
    │   ├── Figures/
    │   │   ├── 1_evaluation_metric.png
    │   │   ├── 2_bleu_score.png
    │   │   ├── 3_accuracy_vs_length.png
    │   │   ├── 4_accuracy_vs_training_size.png
    │   │   ├── 5_syntactic_validity_pie.png
    │   │   ├── 6_experimental_setup.png
    │   │   ├── dataset1.png
    │   │   ├── dataset2.png
    │   │   └── dataset3.png
    │   └── Symbolic Reasoning Report.pdf


## 📊 Key Results

- **F1 Score:** 0.7211
- **BLEU Score:** 0.6904
- **Precision / Recall:** 0.6875 / 0.7582
- **Syntactic Validity:** 72.2%

## 📄 Project Report

For detailed insights and analysis, refer to the **[Symbolic Reasoning Report.pdf](Report/Symbolic%20Reasoning%20Report.pdf)**.