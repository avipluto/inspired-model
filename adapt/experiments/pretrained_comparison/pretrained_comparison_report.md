# Pretrained Model vs. Scratch-Trained Model Comparative Analysis

## 1. Overview
This experiment compares the compact, scratch-trained ADAPT Decision Transformer against a standard pretrained-scale transformer architecture on the identical held-out evaluation test suite (500 scenarios).

---

## 2. Quantitative Comparison Table

| Metric / Dimension | ADAPT Scratch Model | Pretrained Transformer Scale | Ratio / Factor |
| :--- | :---: | :---: | :---: |
| **Model Parameters** | **1,606,724 (~1.6M)** | 44,301,312 (~44.3M) | **27.6x smaller** |
| **Storage / Memory Size** | **18.47 MB** | 169.00 MB | **9.2x smaller** |
| **CPU Latency per Request** | **32.53 ms** | 425.43 ms | **13.1x faster** |
| **Top-1 Decision Accuracy** | **84.00%** | 46.20% | **Task-optimized** |
| **Mean Reciprocal Rank (MRR)** | **0.9112** | 0.6577 | **+0.2535 MRR** |
| **Graded NDCG@4** | **0.9577** | 0.9036 | **+0.0540 NDCG** |

---

## 3. Engineering & Research Analysis

1. **Efficiency and Deployability**:
   - The compact scratch-trained transformer achieves **32.53 ms** per decision on standard CPU hardware with an on-disk footprint of only **18.47 MB**.
   - Standard pretrained LLM architectures require substantial memory and compute (425.43 ms latency, 44.3M params), making them expensive and slow for real-time edge or CPU decision support.

2. **Domain Representation vs. Generic Web Corpus**:
   - Monolithic pretrained language models trained on web corpora lack intrinsic arithmetic grounding and require complex prompt engineering or reinforcement learning to avoid arithmetic hallucination.
   - In contrast, ADAPT's modular architecture delegates arithmetic to deterministic solvers, risks to an empirical MLP, and constraints to hard rejection gates, allowing a compact **1.6M parameter** model to provide reliable candidate ranking without bloat.
