# Product Requirements Document (PRD)

**Project Name:** MIT iQuHack LABS Solver: Combining MTS with VQE, WS-QAOA, and Guided Random Walk 
**Team Name:** 103Qubits
**GitHub Repository:** https://github.com/FelineAlloy/2026-NVIDIA-iQuHack.git  

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Sienna O'Shea | [@osheasienna] | [@siennaoshea] |
| **GPU Acceleration PIC** (Builder) | Maria Neagoie | [@vioneers] | [@maria_n8] |
| **Quality Assurance PIC** (Verifier) | Luxin Matasaru, Utu Kaita | [@felinealloy], [@utukaita] | [@luxin4299], [@utukaita] |
| **Technical Marketing PIC** (Storyteller) | Van Khue Nguyen | [@tieudaochannhan] | [@vankhuenguyen] |

**Team Members (full list):** Luxin Matasaru, Maria Neagoie, Sienna O'Shea, Utu Kaita, Van Khue Nguyen  

---

## 2. The Architecture
**Owner:** Project Lead

### Pipeline Overview
We build a hybrid system where **quantum generates “good starting strings”**, and **classical MTS does the heavy lifting** (deep local improvement + recombination). Our full pipeline is:

1. **VQE Warm-Start Generator**  
   Run many *short* VQE runs (multi-start) to produce a small set of strong candidate strings (starting positions).

2. **Warm-Start QAOA (WS-QAOA) Seed Sampler**  
   Use those VQE candidates as warm-start anchors and sample a larger pool of low-energy-biased bitstrings.

3. **MTS + Canonicalisation + Deduplication**  
   Refine the seed population using a modified MTS where we:
   - canonicalize each string under known symmetries (global flip + reversal),
   - deduplicate to keep population slots “meaningfully different,”
   - keep MTS mechanics intact but accelerate the expensive evaluation loops.

4. **Guided Random Walk Restarts (stagnation escape)**  
   When MTS stagnates, trigger guided random walk sampling to inject new candidates and escape local minima.

### Choice of Quantum Algorithm
* **Algorithm(s):**
  * **Warm-Start Generator:** **VQE**
  * **Seed Sampler:** **Warm-Start QAOA (WS-QAOA)**
  * **Restart / Escape Hatch:** **Guided Random Walk Sampler**

* **Motivation (why these fit LABS + our pipeline):**
  * **VQE**: produces *better-than-random* starting strings fast. We don’t need perfect convergence; we need several “pretty good” anchors that are diverse.
  * **WS-QAOA**: acts like a **seed factory**. It can produce lots of candidate strings at relatively shallow depth, biased toward low energy, which is exactly what we want before the classical search.
  * **Guided Random Walk**: LABS has many local minima. Instead of making quantum do everything, we use it as a controlled “jump generator” when the classical search gets stuck.

### Classical Optimization Core: MTS with Canonicalisation and Deduplication
* **Canonicalisation:** map each candidate sequence to a **canonical representative** under LABS symmetries (**global sign flip** and **sequence reversal**). This prevents multiple equivalent encodings from occupying population slots.
* **Deduplication:** remove duplicate canonical representatives:
  - **before** population construction, and
  - **before** reinjection during restarts.
* **Parallelisation (how we scale MTS without rewriting it into chaos):**
  - **Population-level parallelism:** children in a generation can be processed independently → evaluate and locally improve them in GPU batches.
  - **Tabu/local-search parallelism (candidate lists):** instead of evaluating all `N` one-bit flips one-by-one, evaluate a **candidate list** of `B` flips (e.g., 64 or 128) in a single batched GPU call and pick the best move.
* **Motivation:** canonicalisation + deduplication increase *effective* diversity, reduce wasted evaluation on symmetry copies, and make both quantum seeding and MTS refinement more efficient end-to-end.

### Literature Review
* **Reference:** *Scaling advantage with quantum-enhanced memetic tabu search for LABS* (Cadavid et al., 2025). https://arxiv.org/abs/2511.04553  
  **Relevance:** establishes the core pattern we follow: quantum-generated seeds + classical MTS refinement, with scaling/time-to-solution benchmarking mindset.
* **Reference:** *Warm-starting quantum optimization* (Egger, Mareček, Woerner, 2021). https://quantum-journal.org/papers/q-2021-06-17-479/  
  **Relevance:** motivates why warm-starting can yield more useful low-depth quantum behavior (perfect for seed generation).
* **Reference:** *A Quantum Approximate Optimization Algorithm* (Farhi, Goldstone, Gutmann, 2014). https://arxiv.org/abs/1411.4028  
  **Relevance:** baseline QAOA framework; we adapt it into WS-QAOA as a sampler.
* **Reference:** *A variational eigenvalue solver on a quantum processor* (Peruzzo et al., 2014). https://www.nature.com/articles/ncomms5213  
  **Relevance:** foundational VQE reference supporting our VQE warm-start stage.
* **Reference:** *Guided quantum walk* (Schulz, 2024). https://link.aps.org/doi/10.1103/PhysRevResearch.6.013312  
  **Relevance:** motivates quantum-walk-style exploration for “jumping” to new regions when MTS stagnates.
* **Reference:** *New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search* (Zhang et al., 2025). https://arxiv.org/abs/2504.00987  
  **Relevance:** reinforces that MTS remains strong for large LABS and highlights the value of parallelising the classical core.
* **Reference:** *Evidence of Scaling Advantage for the QAOA on a Classically Intractable Problem* (Shaydulin et al., 2023/2024). https://arxiv.org/abs/2308.02342  
  **Relevance:** domain-relevant QAOA-on-LABS baseline that helps justify hybridizing and warm-starting.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** treat the quantum stage as a **high-throughput sampler** and accelerate it by batching “many small runs,” not by chasing one huge run.
  * **GPU target:** use CUDA-Q’s NVIDIA backend on a single **L4** GPU (primary target).
  * **Batching approach:**
    - **Multi-start parallelism:** run multiple VQE initializations and WS-QAOA parameter sets as an embarrassingly-parallel workload (many independent circuits).
    - **Shot batching:** generate many samples per circuit efficiently (seed factory behavior).
    - **Depth control:** keep circuits shallow (small `p` for QAOA, short VQE budgets) because the classical stage is the real optimizer.
  * **Why this matches L4:** L4 likes throughput. Many medium/small circuits + lots of shots tends to be more cost-effective than “one monster circuit.”

### Classical Acceleration (MTS)
* **Strategy:** GPU-accelerate the **hot loop** in MTS: repeated energy evaluation over large numbers of candidate moves (neighbor flips) during tabu/local search and population refinement.

  * **What we accelerate:** the repeated computation of LABS autocorrelations and energies across many candidate sequences generated inside MTS (mutation, combine, tabu neighborhood scan).

  * **How we accelerate (GPU batching):**
    - Represent sequences as GPU arrays (`cupy.ndarray`) and compute energies for a **batch** of candidates at once rather than one-by-one on CPU.
    - During tabu search, instead of evaluating all `N` single-bit flips sequentially, evaluate a **candidate list** of `B` flips (e.g., 64 or 128) and compute their energies in one GPU call.
    - For population updates, compute energies for all children produced in a generation in a single batch (vectorized GPU computation).

  * **Core technical change (reduce compute per move):**
    - Maintain autocorrelation terms `C_k` and update them incrementally when flipping a bit (reducing full recomputation).
    - Combine this with GPU batching when scanning candidate flips to minimize wall-clock time.
    - **Goal:** replace repeated `O(N^2)` full energy recomputation with lower-cost updates + batched evaluation.

  * **Integration plan:**
    - Implement a GPU-backed energy module `energy_gpu.py` with two entry points:
      1. `energy_batch(S)`: compute energies for a batch of sequences `S` of shape `(B, N)`
      2. `best_flip_batch(s, flip_idx)`: evaluate a candidate list of flips for one sequence
    - Keep a CPU fallback for correctness testing and small-`N` unit tests.

  * **Expected impact:** MTS spends most runtime inside neighbor/energy evaluation. Batching those computations on the GPU should provide the largest speedup per engineering effort on an L4 instance.

### Hardware Targets
* **Dev Environment:** qBraid CPU backend for logic + correctness  
* **Production Environment:** Brev **L4** for GPU acceleration and final benchmarking

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** `pytest`
* **AI Hallucination Guardrails (how we trust AI-generated code):**
  - Tests are written **first** for all kernels/modules (TDD).
  - Any AI-generated optimization must pass:
    - **symmetry properties**,  
    - **CPU vs GPU equivalence checks**,  
    - **small-N brute force ground truth**,  
    before it can be merged.

### Core Correctness Checks
* **Check 1 (Symmetry invariance):**
  - For any sequence `S`, energies must satisfy:  
    `E(S) == E(-S)` and `E(S) == E(reverse(S))`
  - Canonicalisation must preserve energy and be idempotent:  
    `canon(canon(S)) == canon(S)`

* **Check 2 (Ground truth for small N):**
  - For small sizes (e.g., `N <= 10`), brute force all `2^N` sequences on CPU and confirm:
    - energy function matches brute force outputs
    - reported best solution matches known brute-force optimum (energy and/or merit factor)

* **Check 3 (CPU/GPU consistency):**
  - For random batches of sequences, assert:  
    `energy_cpu(S_batch) == energy_gpu(S_batch)` (within exact integer equality if we keep integer arithmetic).

* **Check 4 (Pipeline sanity checks):**
  - Seeds from quantum stages should be **better than random on average**:
    - compare distributions of `E(seed)` vs `E(random)` for same `N`.
  - MTS seeded by WS-QAOA should beat or match MTS seeded randomly under equal wall-clock/iteration budgets.

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Orchestration Strategy:** Human-in-the-loop CI/CD pipeline.
  - **Code Generation:** AI agents generate CUDA-Q kernels and GPU batch-evaluation code from *explicit* Hamiltonian / data-shape prompts.
  - **Verification Loop:** QA Lead runs tests first; failures are pasted back to the agent until tests pass.
  - **Documentation:** maintain a `skills.md` file of CUDA-Q “gotchas” + working code patterns to prevent API hallucinations.

### Success Metrics
* **Metric 1 (Quality — Merit Factor):** achieve **Merit Factor `F >= 8.0`** for lengths up to `N = 60`.
* **Metric 2 (Performance — Time-to-Solution):** achieve **≥ 50× speedup** for the tabu/energy-evaluation phase using CuPy (GPU) vs NumPy (CPU) at `N = 50`.
* **Metric 3 (Scalability):** successfully run the full **Quantum Seed + Classical Optimize** workflow for `N = 100` within **< 5 minutes** wall-clock.

### Visualization Plan
* **Plot 1:** “Quantum Advantage” curve — Time-to-Solution (log scale) vs N, comparing CPU baseline vs GPU-accelerated hybrid.
* **Plot 2:** Search landscape exploration heatmap — show quantum seeding starts closer to low-energy basins vs random starts.
* **Plot 3:** Energy convergence profile — Energy vs iteration showing hybrid converges faster (fewer steps) than standard local search.

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC

* **Plan (don’t burn credits like a sleep-deprived wizard):**
  - Develop and debug on **CPU first** (qBraid) until tests pass.
  - Use Brev **L4** only for:
    - profiling (to find the hot loop),
    - GPU porting (CuPy and CUDA-Q GPU backend),
    - short, time-boxed benchmark runs.
  - Enforce a “stop-the-instance” habit:
    - GPU PIC is responsible for shutting down the Brev instance during breaks and when not actively running jobs.
  - Benchmark with fixed budgets:
    - same number of evaluations / same time cap when comparing methods (fairness + cost control).
