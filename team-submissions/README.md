# 02_ws-qaoa_LABS.ipynb — Hybrid Quantum–Classical LABS Solver  
**(WS-QAOA seeding + GPU-Accelerated Memetic Tabu Search)**

This notebook is our **Phase 2 submission** for the NVIDIA iQuHACK LABS challenge. It implements a **hybrid quantum–classical optimization pipeline** for the Low Autocorrelation Binary Sequences (LABS) problem.

Our core idea is simple and deliberate:

- **Quantum routines** (Warm-Start QAOA + Mini-VQE) are used to generate *biased, high-quality initial samples*.
- These samples are then refined by a **GPU-accelerated Memetic Tabu Search (MTS)**, which performs the main optimization work efficiently at scale.

This notebook is fully self-contained and is intended to be **run directly by judges** to reproduce our results.

---

## Notebook structure (what you will find)

### 1) LABS objective and utilities
This section defines the LABS cost function and symmetry handling:
- `compute_energy(...)`, `compute_energy_cpu(...)`: LABS energy via squared autocorrelations.
- Canonicalization and deduplication:
  - `canonicalize_sequence`
  - `canonical_key`
  - `deduplicate_population`

These utilities ensure correctness and avoid redundant evaluations due to LABS symmetries.

---

### 2) GPU-batched energy evaluation
Energy evaluation is the dominant cost in MTS. We explicitly restructure it into **batch operations**:
- `energy_batch(...)`: batched energy computation (CPU or GPU via CuPy).
- `neighbors_energy_batch(...)`: parallel evaluation of single-bit-flip neighbors.

This is the primary performance optimization in the notebook.

---

### 3) Classical optimizer: Memetic Tabu Search (MTS)
The classical search pipeline consists of:
- Genetic operators:
  - `combine(...)` (crossover)
  - `mutate(...)`
- Local optimization:
  - `tabu_search(...)` using batched neighborhood evaluation
- Full loop:
  - `memetic_tabu_search(...)`

This component refines candidate sequences into low-energy LABS solutions.

---

### 4) Quantum seeding: WS-QAOA + Mini-VQE
This section constructs and executes the quantum sampling stage:
- Mini-VQE components:
  - `vqe_ansatz(...)`
  - `cost_function(...)`
- Warm-start QAOA pipeline:
  - `build_labs_hamiltonian(...)`
  - `get_interactions(...)`
  - `pauli_string_rotation(...)`
  - `trotterized_circuit(...)`
  - `generate_quantum_population(...)`

The output is a **biased initial population** for MTS, outperforming random initialization.

---

### 5) Guided quantum diversification (“quantum kick”)
To escape stagnation, we introduce a controlled diversification mechanism:
- `labs_gqw_kernel(...)`
- `quantum_kick_search(...)`

These routines inject new quantum-generated candidates when classical progress plateaus.

---

### 6) Experiment harness and visualization
This final section handles execution and reporting:
- `run_benchmark(...)`
- `run_full_experiment(...)`
- `visualize_results(...)`
- `plot_labs_histogram(...)`
- `append_jsonl(...)`, `get_system_metadata(...)` for logging results and hardware info

---

## How to run the notebook (judge workflow)

1. **Run all cells from top to bottom**.
   - All functions and configuration switches are defined before execution blocks.
2. Select runtime settings in the **configuration section**:
   - CPU vs GPU
   - precision
3. Execute the provided experiment cells.
   - The notebook will automatically:
     - generate quantum samples
     - run MTS refinement
     - log results
     - produce plots

No external scripts are required.

---

## Configuration options (important)

### Backend and precision
Defined near the top of the notebook:
- `MODE = "auto"`  
  Options: `"auto" | "cpu" | "gpu" | "mgpu"`
- `PRECISION = "fp32"`  
  Options: `"fp32" | "fp64"`

Interpretation:
- **auto**: GPU if available, otherwise CPU.
- **gpu**: force single-GPU execution.
- **mgpu**: multi-GPU execution (useful only for larger runs).
- **fp32**: default and recommended for performance.
- **fp64**: higher numerical precision, slower.

---

## Expected outputs

When the notebook is run successfully, it produces:
- The **best LABS sequence** found (±1 representation).
- Its corresponding **energy**.
- Histograms and plots comparing:
  - quantum samples
  - initial populations
  - post-MTS refined populations
- A **JSONL log file** containing:
  - problem size
  - runtime configuration
  - timing and evaluation counts
  - system and GPU metadata

---

## Correctness and validation checks

We explicitly verify correctness throughout the notebook via:

1. **LABS symmetry checks**
   - Global flip invariance: `E(s) = E(-s)`
   - Reversal invariance: `E(s) = E(s[::-1])`

2. **CPU vs GPU consistency**
   - Batched GPU energy results are cross-checked against CPU reference computations.

3. **Neighborhood evaluation sanity**
   - GPU-batched neighbor ranking matches brute-force CPU results.

4. **Deduplication correctness**
   - Symmetry-equivalent sequences map to the same canonical form.

5. **Small-N ground truth**
   - For small N, results are validated against brute-force optima.

These checks ensure both **numerical correctness** and **algorithmic validity**.

---

## Notes on GPU execution (Brev)

The notebook is compatible with Brev-based GPU environments.  
GPU usage is automatic when available and requires no code changes.

---

## Role in the iQuHACK submission

This notebook represents our **Phase 2 final implementation**, demonstrating:
- a hybrid quantum–classical workflow,
- GPU acceleration of the classical core,
- and systematic validation.

It is designed to be the **primary artifact** for evaluating our technical contribution.

---

## References

- NVIDIA iQuHACK LABS challenge specification and milestones  
- QE-MTS / DCQO literature motivating quantum-assisted seeding  
- LABS problem symmetry and optimization background
