# ADAPT: Adaptive Decision and Planning Transformer
## System Architecture & Engineering Specification

### 1. Executive Summary & Design Mission
ADAPT (Adaptive Decision and Planning Transformer) is an experimental decision-support system designed to evaluate candidate actions under multi-criteria constraints, uncertainty, and user preferences. Unlike monolithic black-box conversational LLMs that attempt to synthesize arithmetic, domain rules, and action rankings in a single hallucination-prone prompt, ADAPT employs a modular, four-pillar architecture:

1. **Compact Decision Transformer (Component A)**: A decoder-only transformer trained from random initialization on structured decision contexts and action tokens to learn candidate-action scoring representations.
2. **Structured Consequence Predictor (Component B)**: A specialized PyTorch multi-layer perceptron (MLP) trained to estimate empirical, non-deterministic continuous and categorical outcomes (e.g., commute fatigue, burnout risk, unexpected ancillary costs).
3. **Deterministic Decision Aggregator & Policy (Component C)**: An auditable rules, constraint, arithmetic, and trade-off engine. It performs exact arithmetic (e.g., net disposable income = stipend - rent - transport), checks verified hard constraints (e.g., max budget, commute duration limits), triggers clarification when critical information is missing, and abstains when uncertainty or selective risk exceeds thresholds.
4. **Governed Feedback & Learning Pipeline (Component D)**: An offline feedback ingestion and retraining loop with schema validation, provenance verification, and regression evaluation against frozen test suites—preventing uncontrolled online weight updates.

---

### 2. Hardware Environment & Compute Constraints
Following environment inspection on the target host:
- **Operating System**: Windows 11 Home Single Language (x86_64)
- **PyTorch Device**: CPU only (`torch.cuda.is_available() == False`, `torch 2.14.0+cpu`)
- **System Memory**: 16.4 GB total visible RAM, ~3.9 GB free physical RAM
- **Available Storage**: ~21.7 GB free on volume `C:`
- **Design Implications**:
  - The scratch transformer is configured to a compact parameter count (~3.5M parameters, $d_{model}=192$, 6 layers, 6 attention heads, context window $T=256$, vocab size 2048) enabling full convergence on multi-core CPU in minutes without thrashing RAM or exceeding memory limits.
  - The structured predictor MLP is designed with 3 layers ($64 \to 32 \to 16$) training in seconds.
  - All dataset pipelines, feature scalers, tokenizers, and model checkpoints are stored locally in reproducible formats (`.json`, `.pt`, `.jsonl`).

---

### 3. Component Deep Dive

```
+----------------------------------------------------------------------------------------------------+
|                                         ADAPT ARCHITECTURE                                         |
+----------------------------------------------------------------------------------------------------+
                                      User Request / Scenario
                                                 |
                                                 v
                      +------------------------------------------------------+
                      |               FastAPI Ingestion Layer               |
                      +------------------------------------------------------+
                                                 |
                       +-------------------------+--------------------------+
                       |                                                    |
                       v                                                    v
          [Natural Language Scenario]                              [Structured Facts]
                       |                                                    |
                       v                                                    v
       +-------------------------------+                    +-------------------------------+
       |    Component A: Transformer   |                    | Component B: Consequence MLP  |
       |  - Custom Byte-Level BPE      |                    |  - Robust Numerical Imputation|
       |  - Causal Self-Attention      |                    |  - Categorical Embeddings     |
       |  - Log-Likelihood / Ranking   |                    |  - Outcome & Risk Estimation  |
       +-------------------------------+                    +-------------------------------+
                       |                                                    |
                       | Candidate Prior Scores                             | Predicted Risk / Fatigue
                       +-------------------------+--------------------------+
                                                 |
                                                 v
                               +------------------------------------+
                               | Component C: Decision Aggregator   |
                               |  1. Deterministic Financial Math   |
                               |  2. Hard Constraint Verification   |
                               |  3. Trade-off Scoring Engine       |
                               |  4. Uncertainty & Missing Checks   |
                               |  5. Abstention & Clarification     |
                               +------------------------------------+
                                                 |
                                                 v
                                    Audited Recommendation Output
                                   (Decision, Trade-offs, Warnings)
                                                 |
                                                 v
                               +------------------------------------+
                               | Component D: Feedback Pipeline     |
                               |  - Provenance & Verification Gate  |
                               |  - Staged Dataset Storage          |
                               |  - Offline Retraining & Regression |
                               +------------------------------------+
```

#### Component A: Compact Decision Language Model
- **Architecture**: Decoder-only causal transformer (`nn.Module`).
- **Layers**: 6 transformer blocks with pre-layer normalization (`RMSNorm` or `LayerNorm`), multi-head causal self-attention (`MHA`), rotary or learned positional embeddings, and SwiGLU / GELU feed-forward networks (`d_ff = 4 * d_model`).
- **Objective**: Candidate action likelihood scoring $P(a_i | \text{scenario})$. For candidate action tokens, the model computes normalized log-probabilities over candidate actions given the scenario prefix.
- **Tokenizer**: Custom BPE tokenizer trained on decision corpora using Hugging Face's `tokenizers` library with a vocabulary of 2,048 tokens and strict special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`, `<action>`).

#### Component B: Structured Consequence Predictor
- **Architecture**: Multi-layer Perceptron (MLP) with numerical standardization, one-hot/embedding categorical encoding, and dropout for uncertainty estimation (MC Dropout).
- **Inputs**: Numerical features (monthly stipend, base expenses, transit time, rent quote, work-from-home days) and categorical features (internship field, city cost tier, commute mode).
- **Outputs**: Multi-task continuous predictions:
  1. Commute fatigue index $[0, 10]$
  2. Professional burnout risk $[0, 1]$
  3. Estimated unexpected incidental variance $[0, 500]$ USD
- **Loss**: Smooth L1 / Huber loss and Binary Cross Entropy with class-weighting.

#### Component C: Decision Aggregator & Verification Engine
- **Deterministic Arithmetic**:
  $$\text{Net Remaining Funds} = \text{Stipend} - (\text{Rent} + \text{Commute Cost} + \text{Base Living Expenses})$$
  Arithmetic is strictly computed deterministically, never hallucinated by neural components.
- **Hard Constraint Enforcement**:
  - If $\text{Net Remaining Funds} < \text{Minimum Budget Reserve}$, action is marked infeasible (`VIOLATES_BUDGET`).
  - If $\text{Daily Commute Minutes} > \text{Max Acceptable Commute}$, action is marked infeasible (`EXCEEDS_MAX_COMMUTE`).
  - If hybrid work is not offered by employer and user chooses negotiate hybrid with zero policy flexibility, flag risk.
- **Uncertainty & Clarification Policy**:
  - Missing essential facts (e.g., unspecified rent in target city, undefined stipend, or undisclosed commute duration) immediately trigger a `REQUEST_CLARIFICATION` state rather than making blind assumptions.
  - If top candidate action scores have high entropy or conflicting trade-offs with low data support, the aggregator flags `ABSTAIN_UNCERTAIN` or requests human review.

#### Component D: Feedback and Governed Learning Pipeline
- **Validation**: Incoming feedback must adhere to strict Pydantic schemas specifying scenario hash, model version, action selected, user satisfaction rating, and verified empirical outcome.
- **Safety Gate**: Feedback entries are stored in a quarantined staging buffer. No automated weight update occurs online.
- **Offline Retraining**: Retraining is invoked via batch scripts (`retrain_pipeline.py`). Retraining requires passing a strict regression evaluation suite on a frozen golden test set before any new checkpoint version is tagged for production deployment.

---

### 4. Target Use Case: Career & Internship Decision Support
The primary demonstration use case is a structured real-world internship dilemma:
1. **Move / Relocate**: Rent an apartment near the internship site (high cost, near-zero commute, high independence, budget strain).
2. **Commute**: Stay in existing housing/family home and commute (zero or low rent, high commute time, fatigue risk).
3. **Negotiate Hybrid Work**: Propose 2 days in-office and 3 days remote (reduced commute, low rent, professional coordination requirement).
4. **Request Information**: Defer decision until missing stipend details, housing stipends, or commute subsidies are clarified.

---

### 5. Baselines and Evaluation Metrics
- **Baselines**:
  1. Uniform Random Baseline
  2. Deterministic Weighted Multi-Criteria Rule Baseline (AHP/TOPSIS-inspired)
  3. Structured Predictor Alone
  4. Pretrained Transformer Comparison (TinyStories / SmolLM / Qwen-0.5B via LoRA in Phase 10)
- **Metrics**:
  - Top-1 Choice Accuracy
  - Normalized Discounted Cumulative Gain (NDCG@K)
  - Mean Reciprocal Rank (MRR)
  - Mean Absolute Error (MAE) for financial and consequence predictions
  - Constraint Violation Rate (target: 0.0% for Aggregator)
  - Clarification Sensitivity and Specificity
