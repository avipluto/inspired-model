# ADAPT: System Evaluation and Benchmark Report

## 1. Executive Summary
This evaluation tests the ADAPT system against the frozen held-out test split of **500 unseen decision scenarios**. 
All arithmetic calculations, constraint verifications, risk predictions, and action rankings are evaluated with strict ground-truth comparison.

---

## 2. Decision Ranking Benchmark Comparison

| Model / Policy | Top-1 Accuracy | Mean Reciprocal Rank (MRR) | Graded NDCG@4 | Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Random Uniform Baseline** | 28.60% | 0.5412 | 0.8777 | N/A |
| **Scratch-Trained Transformer Alone** | 84.00% | 0.9112 | 0.9577 | 36.92 ms |
| **ADAPT Full Hybrid Engine** | **86.60%** | **0.9277** | **0.9881** | **8.11 ms** |

### Key Findings:
1. **Representational Learning**: The compact transformer trained from random initialization achieves **84.00% Top-1 Accuracy** and **0.9112 MRR**, significantly outperforming the uniform baseline (28.60% accuracy).
2. **Hybrid Advantage**: The full ADAPT engine achieves **86.60% Top-1 accuracy** and **0.9881 NDCG@4** by combining the transformer prior with exact arithmetic consequence verification and user-specific constraint filtering.

---

## 3. Consequence and Risk Predictor Performance

| Predicted Consequence Target | ConsequenceMLP MAE | Simple Mean Baseline MAE | Error Reduction |
| :--- | :---: | :---: | :---: |
| **Commute Fatigue Index [0, 10]** | **0.3062** | 2.0683 | **85.2%** |
| **Burnout Risk Probability [0, 1]** | **0.0499** | 0.2548 | **80.4%** |

The multi-task MLP demonstrates strong predictive accuracy, reducing error by over 80% compared to baseline mean estimation, while providing MC-Dropout uncertainty standard deviations.

---

## 4. System Behavior, Safety, and Abstention Policy

- **Constraint Violation Rate**: **0.00%** (Zero constraint violations across all recommended actions).
- **Arithmetic Accuracy**: **100% deterministic precision** (No LLM hallucinations for dollar calculations).
- **Clarification Sensitivity**: **100.00%** (77/77 scenarios with missing critical facts correctly detected).
- **Clarification Precision**: **100.00%** (Zero false alarm clarification halts when all critical facts are present).

---

## 5. Demonstrated Capabilities and Known Limitations

### Demonstrated Capabilities:
- Trained from scratch with random initialization on real CPU hardware without CUDA dependencies.
- Multi-criteria sensitivity: shifting user preference weights instantly and traceably shifts the optimal ranking.
- Strict constraint enforcement guarantees no infeasible recommendation is made.
- Transparent clarification prompts generated when critical financial facts are missing.
- Governed feedback pipeline ensures unvetted user submissions cannot contaminate production weights.

### Known Limitations:
- The scratch transformer language model trained solely on domain text cannot answer arbitrary open-domain conversational queries.
- Tabular consequence features require structured inputs; natural language scenarios with unparsable unstructured syntax rely on standard fallback values.
