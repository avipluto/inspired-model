# Dataset Provenance & Data Governance

## 1. Dataset Origin & Classification
- **Dataset Name**: `adapt_internship_decision_v1`
- **Classification**: Transparent Synthetic Rule-Grounded Dataset with Verified Arithmetic.
- **License**: CC-BY-4.0 (Original creation for the ADAPT experimental research project).
- **Generator**: `adapt/scripts/generate_dataset.py`
- **Purpose**: Benchmarking decision transformers, consequence predictors, and constraint engines on multi-criteria career logistics (relocation vs commuting vs hybrid negotiation vs clarification).

## 2. Integrity and Labeling Rules
- **Explicit Labeling**: All generated records are stamped with `provenance: "synthetic_rule_grounded_v1"`.
- **No Hallucinated Ground Truth**:
  - Financial calculations (Net Remaining Funds) are computed with deterministic arithmetic:
    - `MOVE`: $\text{stipend} - (\text{relocation\_rent} + \text{base\_living\_expenses})$
    - `COMMUTE`: $\text{stipend} - (\text{commute\_monthly\_cost} + \text{base\_living\_expenses})$
    - `NEGOTIATE_HYBRID`: $\text{stipend} - (\text{commute\_monthly\_cost} \times \frac{2}{5} + \text{base\_living\_expenses})$
    - `REQUEST_INFO`: $0$ change (status quo).
  - Feasibility is determined by checking verified user constraints (`min_savings_buffer`, `max_acceptable_commute_minutes`, and employer remote flexibility).
  - Scenarios with missing essential financial or commute information designate `REQUEST_INFO` as the ground-truth optimal choice (`should_clarify = True`).
- **Data Splitting**:
  - Train: 1,600 scenarios.
  - Validation: 400 scenarios.
  - Test: 500 held-out scenarios generated from distinct parameter distributions to eliminate near-duplicate leakage.
- **External Public Corpora**:
  - The custom BPE tokenizer is trained on a combined corpus of curated structured decision narratives and licensed decision descriptions (Apache-2.0 / CC-BY-4.0).
