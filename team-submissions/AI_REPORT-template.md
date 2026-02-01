# AI Usage Report

> **Note:** The questions below are used as prompts to guide reflection. The goal of this report is to demonstrate that AI agents were employed thoughtfully, with human oversight and verification at every stage.

---

## 1. The Workflow  
**How did you organize your AI agents?**

We organized our use of AI agents by assigning them clearly scoped roles that complemented human decision-making rather than replacing it.

- We primarily used **ChatGPT and Gemini** as *boosted search engines* to:
  - Compare different algorithmic approaches
  - Explore alternative design choices
  - Assist with writing long, repetitive, or boilerplate-heavy code
- AI was frequently used to **explain quantum computing concepts**, including QAOA, sampling behavior, and Trotterization, which accelerated onboarding and collective understanding.
- We used our **CUDA-Q / Coda credits** specifically for CUDA-Q kernel structure.

This organization allowed us to offload low-level or explanatory tasks to AI, while preserving human focus for algorithmic reasoning, experimental design, and performance tuning.

## 2. Verification Strategy  
**How did you validate code created by AI?**

We did not treat AI-generated code as production-ready. Every AI-assisted contribution passed through both **human review** and **unit testing**.

### Human Review 

- All team members worked together at a shared table
- Each contributor presented their sections of code, and logic and assumptions were explained and challenged
- AI-generated code was always rethought and often rewritten before inclusion

This ensured that no hallucinated logic or failures were added to the codebase.

### Unit Tests

After human review, we validated correctness through targeted unit tests designed to catch AI hallucinations or logic errors.

**Examples of unit tests we used:**

- **Small-Instance Quantum Sampling Tests**
  - Ran sampling on very small problem sizes (e.g., `N = 4 to 8`)
  - Verified:
    - Sampling terminated correctly
    - Returned bitstrings were valid
    - Energy values matched brute-force or known results

- **Stability & Consistency Checks**
  - Repeated sampling runs with fixed seeds
  - Ensured distributions and best-energy states were stable

- **Backend Sanity Checks**
  - Compared CPU and GPU executions using identical parameters
  - Verified that qualitative behavior (ordering of energies, convergence trends) was consistent

These tests ensured that AI-generated code could not silently introduce incorrect logic.

## 3. The "Vibe" Log

### Win  
**One instance where AI saved you hours**

AI saved us several hours during infrastructure setup, particularly when:
- Switching between CPU and GPU backends
- Selecting the number of GPUs
- Configuring CUDA-Q execution modes
- Implementing quantum / MTS control logic

These tasks involved long, repetitive, low-risk code that was easy to reason about but time-consuming to write manually. AI handled this efficiently and reliably.


### Learn  
**One instance where you altered your prompting strategy to get better results**

We observed a significant improvement in AI output quality once we:
- Provided full notebook context
- Clearly explained our algorithmic intent
- Specified compute-credit constraints
- Highlighted known caveats and performance bottlenecks

By shifting from short prompts to **context-rich prompts**, AI responses became more precise and relevant for our workflow.


### Fail  
**One instance where AI failed or hallucinated, and how you fixed it**

AI struggled when asked to design a **random walk–based method**, suggesting parameters that:
- Failed to converge
- Performed poorly in practice
- Lacked theoretical justification

We fixed this by:
- Abandoning the AI-suggested parameters
- Re-deriving the method manually
- Tuning parameters empirically through controlled experiments

This reinforced our approach of keeping our team members responsible for correctness and theoretical soundness.

### Context Dump  
**Prompts, `skills.md`, MCP, or other artifacts demonstrating thoughtful prompting**

Throughout the project, we provided AI with:
- Full Jupyter notebooks
- Partial implementations
- Detailed descriptions of our ideas
- Explicit compute and time constraints
- Warnings about known failure modes

This deliberate context-sharing significantly reduced hallucinations and improved the relevance of AI assistance. One example could be: 

"Please find the notebook attached. I implemented a `run_full_experiment` wrapper that puts together the quantum sampling stage, the classical refinement stage, and a light benchmarking layer. We also added explicit timing splits (quantum vs. classical) and JSON-based logging that records configuration parameters, hardware/backend metadata, and outcome quality metrics.

Please review **only** the benchmarking and logging components (i.e., the `run_full_experiment` function and the associated metadata/JSON helpers) and assess whether their structure and metric choices are appropriate for fair comparison. I'm especially interested in whether the selected metrics (wall-clock timing, throughput, and solution quality indicators) are sufficient and well-justified, and if you can please come up with recommendations for refinements or additional metrics."

## Summary

AI was used as a **force multiplier** and all outputs were filtered through human reasoning and validated via unit tests, allowing us to move faster without compromising scientific rigor.
