# Product Requirements Document (PRD)

**Project Name:** MIT iQuHack LABS Solver: Combining MTS with VQE, WS-QAOA, and Guided Random Walk <br>
**Team Name:** 103Qubits <br>
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

### 2. Pipeline Overview
We implement a hybrid pipeline where quantum routines generate high-quality and diverse initial candidates, and a classical Memetic Tabu Search (MTS) performs the main optimization. The pipeline is:

1. **VQE Warm-Start Generator:** run short, parallel/multi-start VQE to produce a small set of strong candidate strings (starting positions).
2. **WS-QAOA Seed Sampler:** use those candidates as warm starts for WS-QAOA to generate a large pool of biased, low-energy bitstring samples.
3. **MTS Refinement (with canonicalisation + deduplication):** canonicalise seeds to remove symmetry-equivalent duplicates, deduplicate the seed pool, then run MTS as the primary refinement engine.
4. **Guided Random Walk Restarts:** detect stagnation (no improvement over a fixed window) and inject new candidates produced by a guided random walk sampler to escape local minima, followed by canonicalisation and reinsertion into the population.

---

### Choice of Quantum Algorithms

- **Algorithm 1 (Warm-Start Generator): VQE**
  - **Role in pipeline:** VQE is used to generate a small set of strong *starting positions* (candidate sequences) that serve as warm-start anchors.
  - **Motivation:** VQE can efficiently refine parameters toward low-energy configurations, producing higher-quality starting points than purely random initialisation. We run multiple short VQE instances (multi-start) to obtain diverse candidates rather than relying on a single converged solution.

- **Algorithm 2 (Seed Sampler): Warm-Start QAOA (WS-QAOA)**
  - **Role in pipeline:** WS-QAOA uses VQE-produced candidates as warm starts and generates a large set of measurement samples (bitstrings) that are biased toward low energy.
  - **Motivation:** Our objective is not to solve LABS purely with quantum, but to produce a *seed population* that improves time-to-solution for MTS. WS-QAOA is suited for this because it can produce many candidate bitstrings at shallow depth, which is practical under GPU simulation constraints.

- **Algorithm 3 (Stagnation Escape): Guided Random Walk Sampler**
  - **Role in pipeline:** When MTS stagnates, we trigger a guided random walk procedure to generate new candidate strings for reinjection into the population.
  - **Motivation:** LABS has a rugged energy landscape with many local minima. A guided random walk provides non-local exploration and supports controlled restarts without replacing the fast inner-loop mechanics of MTS.

---

### Classical Optimization Core: MTS with Canonicalisation, Deduplication, and Parallelisation

- **Canonicalisation:** we map each candidate sequence to a canonical representative under known LABS symmetries (global sign flip and sequence reversal) to prevent multiple equivalent encodings from occupying population slots.

- **Deduplication:** we remove duplicate canonical representatives before population construction and before reinjection during restarts.

- **Parallelisation:** we parallelise MTS at the *batch level* to exploit GPU throughput while keeping the MTS logic unchanged.
  - **Population-level parallelism:** each generation produces multiple children (via combine/mutation). Their energy evaluation and initial local-improvement steps are independent, so we process children in batches on the GPU.
  - **Tabu/local-search parallelism (candidate lists):** within tabu search, instead of checking all \(N\) single-bit flips sequentially, we evaluate a candidate list of \(B\) flips (e.g., \(B=64\) or \(128\)) in parallel by computing the energies of all \(B\) neighbors in one batched GPU call. The best admissible move is selected and applied.
  - **Energy-evaluation batching:** all repeated objective evaluations (children energies, neighbor energies, restart-injected seed energies) are computed as large matrix operations on the GPU rather than Python loops.

- **Motivation:** canonicalisation + deduplication increase *effective* population diversity and prevent wasted compute on symmetry-equivalent solutions, while parallelisation ensures that the most expensive operation in MTS (repeated energy evaluation across many candidates) is executed efficiently on the GPU. Together, these changes reduce wasted evaluations and improve end-to-end time-to-solution.

---

### Literature Review

- **Reference:** *Scaling advantage with quantum-enhanced memetic tabu search for LABS* (A. G. Cadavid, P. Chandarana, S. V. Romero, J. Trautmann, E. Solano, T. L. Patti, N. N. Hegade, 2025). https://arxiv.org/abs/2511.04553  
  **Relevance:**
  - Establishes the core hybrid pattern we build on: **quantum-generated seeds** + **classical MTS refinement**.
  - Motivates why **non-local sampling** (quantum) helps on LABS’ rugged landscape, while **MTS** does the heavy local improvement.
  - Provides a benchmark mindset (time-to-solution / scaling) that we can reuse for fair comparisons.

- **Reference:** *Warm-starting quantum optimization* (D. J. Egger, J. Mareček, S. Woerner, 2021). https://quantum-journal.org/papers/q-2021-06-17-479/  
  **Relevance:**
  - Justifies the idea behind our pipeline: don’t start quantum search from “pure randomness” if we can start from a **decent classical guess**.
  - Supports using warm-starts to get **useful low-depth performance**, which matters when we need many samples (seed factory behavior).

- **Reference:** *A Quantum Approximate Optimization Algorithm* (E. Farhi, J. Goldstone, S. Gutmann, 2014). https://arxiv.org/abs/1411.4028  
  **Relevance:**
  - Provides the baseline QAOA framework that WS-QAOA modifies.
  - Gives the simplest conceptual model for “layered quantum improvement + mixing,” which we adapt into a seed generator rather than a standalone solver.

- **Reference:** *A variational eigenvalue solver on a quantum processor* (A. Peruzzo et al., 2014). https://www.nature.com/articles/ncomms5213  
  **Relevance:**
  - Foundational VQE reference motivating our **VQE warm-start generator** stage.
  - Supports the design pattern “short variational runs can find a good region,” which we then use as a starting anchor for WS-QAOA sampling.

- **Reference:** *Guided quantum walk* (S. Schulz, 2024). https://link.aps.org/doi/10.1103/PhysRevResearch.6.013312  
  **Relevance:**
  - Motivates our **restart sampler**: use a quantum-walk-style exploration method to **jump to new regions** when MTS stagnates.
  - Matches our “quantum as exploration / escape hatch” role (non-local moves), rather than putting quantum inside every MTS step.

- **Reference:** *New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search* (Z. Zhang, J. Shen, N. Kumar, M. Pistoia, 2025). https://arxiv.org/abs/2504.00987  
  **Relevance:**
  - Supports our “GPU-first” mindset: most runtime is in classical search, so accelerating MTS components is a major performance lever.
  - Reinforces that MTS-style heuristics remain state-of-the-art for large LABS instances, so improving MTS efficiency directly improves end-to-end results.

- **Reference:** *Evidence of Scaling Advantage for the Quantum Approximate Optimization Algorithm on a Classically Intractable Problem* (R. Shaydulin et al., 2023). https://arxiv.org/abs/2308.02342  
  **Relevance:**
  - Directly studies QAOA on **LABS**, giving us a domain-relevant baseline for what “plain QAOA” can do.
  - Helps motivate why we treat QAOA as a component (seed generator) and why warm-starting / hybridization may be beneficial.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)

#### Strategy Overview

Our quantum component is dominated by repeated execution and sampling of parameterized quantum circuits (DSWQA / Trotterized QAOA-like evolution) using **statevector-based simulation** in CUDA-Q. The computational cost scales exponentially with the number of qubits $N$, both in **memory** $O(2^N)$ and **compute** $O(\text{depth} \cdot 2^N)$, making GPU selection, memory management, and parallelization strategy critical.

Our approach prioritizes **performance per dollar of credits**, **iteration speed**, and **scalability**, rather than raw peak throughput.

#### GPU Selection Rationale (L4 vs T4 vs A100 vs H100)

We explicitly evaluated available NVIDIA accelerators in the context of **short, repeated simulation jobs**, not long-running training workloads.

**L4 (Ada Lovelace, 24 GB VRAM)** is our primary target:
- Sufficient VRAM to support statevector simulation up to moderate $N$;
- Strong FP16 / Tensor throughput well-suited to gate-dense circuit simulation;
- Very good performance-per-credit, enabling approximately **20–30 GPU-hours** within a 20$ Brev credit budget;
- Ideal for parameter sweeps, schedule exploration, and repeated sampling runs.

**T4 (Turing, 16 GB VRAM)** is acceptable only for early CPU-offloaded debugging:
- Smaller memory footprint restricts feasible $N$;
- Lower throughput increases iteration latency;
- Used only as a fallback development option.

**A100 (Ampere, 40–80 GB VRAM)** is considered a production-only option:
- Large VRAM enables single-GPU simulation at higher $N$;
- Significantly higher cost per hour, making it unsuitable for iterative tuning under tight credit constraints.

**H100 (Hopper)** is excluded:
- Its cost-to-benefit ratio is unfavorable for our challenge, making it unnecessary overkill.

Overall, for our team’s workload and budget profile, **L4 provides the best acceleration-per-credit**. The **A100** is reserved only for final benchmark demonstrations if required.

#### Concrete GPU Execution Plan

##### Phase 1 — Single-GPU L4 (Primary Development)

- **Backend:** CUDA-Q single-GPU statevector simulator on Brev L4  
- **Purpose:** Parameter tuning, correctness validation, and exploratory sweeps  

This phase maximizes **scientific output per GPU-hour** by enabling rapid iteration under tight credit constraints.

##### Phase 2 — Multi-GPU Scaling

Once single-GPU memory limits are reached, we target **distributed statevector simulation**.

- **Backend:** `nvidia-mgpu` (CUDA-Q)
- **Strategy:**
    - Shard the statevector across multiple L4 GPUs;
    - Use qubit ordering and circuit structure to minimize cross-GPU communication;
    - Favor gate patterns that reduce global synchronization overhead.

This phase enables exploration of **larger $N$** without switching to high-cost GPUs.

##### Phase 3 — Final Benchmarks (Production)

Depending on backend stability and cost efficiency:

- **Preferred:** Multi-L4 `mgpu` backend (better cost scaling);
- **Fallback:** Single **A100-80GB** GPU for maximum $N$ on one device.

This phase is **strictly limited to final demonstrations and result plots**.

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
**Dev Environment:**  
qBraid CPU backend for algorithmic logic, kernel correctness, and small $N$ self-validation, Brev L4 for initial GPU testing. 

**Production Environment:**  
Brev **L4** GPU backend for CUDA-Q acceleration, large $N$ sampling experiments, and final benchmark runs, with multi-L4 scaling via the `nvidia-mgpu` backend when available. We will also look into the possibility of turning to Brev A100-80GB.

---

## 4. Verification & Validation Plan
**Owner:** Quality Assurance PIC
### Correctness & Consistency
- **Unit testing (TDD):** kernels/modules are test-driven using `pytest`. AI-generated code is merged after passing symmetry, and small-N ground-truth tests.
- **Symmetry invariance:** For any sequence `S`, energies must satisfy:  `E(S) == E(-S)` and `E(S) == E(reverse(S))`
- **Canonicalisation is idempotent:** `canon(canon(S)) == canon(S)`
- **Small-N ground truth:** For $N < 10$, brute-force enumeration of all $2^N$ sequences matches energy values and known optima.
### Quantum Seed Validation
 WS-QAOA energy histograms are visibly shifted toward lower energies compared to random sampling.
### Performance & Scaling
Time to solution vs N plot shows a shallower slope for the hybrid solver than the baseline.
### Reproducibility
All runs use fixed random seeds, committed scripts, and record N, wall-clock time, evaluation count, and hardware backend.

---

## 5. Execution Strategy & Success Metrics  
**Owner:** Technical Marketing PIC  

### Agentic Workflow
- **Orchestration Strategy:** We implement a “Human-in-the-loop” CI/CD pipeline.
  - *Code Generation:* AI agents (via Gemini Pro 3.0 / Claude 3.5 Sonnet) generate CUDA-Q kernels based on specific physics Hamiltonian prompts.
  - *Verification Loop:* The QA Lead writes unit tests *first* (TDD). Failing logs are fed back to the agent to iteratively refactor the code until all tests pass.
  - *Documentation:* We maintain a live `skills.md` file containing the latest CUDA-Q API documentation to prevent agent hallucinations.

### Success Metrics
- **Metric 1 (Quality — Merit Factor Focus):** Achieve a **Merit Factor** \(F\) **≥ 8.0** for sequence lengths up to **\(N = 60\)**. This ensures our hybrid solver produces sequences with high “energy efficiency” relative to their length, a key requirement in signal processing.
- **Metric 2 (Performance — Time-to-Solution):** Achieve at least **50× speedup** for the **TABU search phase** using `CuPy` (GPU) parallel evaluation compared to the baseline `NumPy` (CPU) implementation for **\(N = 50\)**.
- **Metric 3 (Scalability):** Successfully execute the **Quantum Seed + Classical Optimize** workflow for **\(N = 100\)** within the strict time limit of **under 5 minutes**.

### Visualization Plan
- **Plot 1: The “Quantum Advantage” Curve:** A dual-axis line chart showing *Time-to-Solution (log scale)* vs *Problem Size \(N\)*. One line for CPU baseline, one for GPU-accelerated. The widening gap demonstrates our success.
- **Plot 2: Landscape Exploration Heatmap:** A 2D visualization of the search space, highlighting how the *Quantum Seed* lands the solver closer to the global minimum (low energy valley) compared to a random start.
- **Plot 3: Energy Convergence Profile:** A plot of \(E(S)\) vs iterations, proving that our Hybrid MTS converges to the optimal solution with fewer steps than standard local search.

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
