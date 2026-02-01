import numpy as np
import matplotlib.pyplot as plt

### CANONICALIZATION SANITY CHECK

def run_canonical_sanity_check(canonical_key_func, canonicalize_sequence_func, N=20):
    """
    Validates that the provided canonicalization utilities follow the 
    invariants required for the LABS MTS algorithm.
    """
    print(f"Starting Canonicalisation sanity check (N={N})...")
    
    # Generate test sequence
    np.random.seed(123)
    s = np.random.choice([-1, 1], size=N).astype(np.int8)

    try:
        # Test 1: Inversion Symmetry (s and -s)
        assert canonical_key_func(s) == canonical_key_func(-s), \
            "FAIL: s and -s should have same canonical key"

        # Test 2: Reflection Symmetry (s and s[::-1])
        assert canonical_key_func(s) == canonical_key_func(s[::-1]), \
            "FAIL: s and s[::-1] should have same canonical key"

        # Test 3: Combined Symmetry (s and -(s[::-1]))
        assert canonical_key_func(s) == canonical_key_func(-s[::-1]), \
            "FAIL: s and -(s[::-1]) should have same canonical key"

        # Test 4: Type and Dtype check
        canon = canonicalize_sequence_func(s)
        assert isinstance(canon, np.ndarray), \
            "FAIL: canonicalize_sequence should return numpy array"
        assert canon.dtype == np.int8, \
            f"FAIL: expected int8, got {canon.dtype}"
        assert len(canon) == N, \
            f"FAIL: expected length {N}, got {len(canon)}"

        # Test 5: Lexicographical Minimality
        # We check all 4 equivalent forms in the LABS symmetry group
        variants = [
            tuple(s.tolist()), 
            tuple((-s).tolist()), 
            tuple(s[::-1].tolist()), 
            tuple((-s[::-1]).tolist())
        ]
        assert tuple(canon.tolist()) == min(variants), \
            "FAIL: canonical form should be lex-smallest of the 4 symmetry variants"

        print("Canonicalisation sanity check: PASS")
        return True

    except AssertionError as e:
        print(f"Check failed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during sanity check: {e}")
        return False


### TABU SEARCH SANITY CHECK
def run_tabu_sanity_check(tabu_search_func, energy_batch_func, to_cpu_func, xp_module, N=20):
    """
    Validates the Tabu Search implementation for the LABS problem.
    Checks for crashes, correct return types, and energy improvement.
    """
    print(f"Starting Tabu Search sanity check (N={N}, seed=42)...")
    
    try:
        # 1. Setup reproducible test case
        np.random.seed(42)
        test_seq = np.random.choice([-1, 1], size=N)
        
        # 2. Get Initial Energy
        initial_energy = int(to_cpu_func(energy_batch_func(xp_module.asarray([test_seq])))[0])
        
        # 3. Run Tabu Search
        # Using a small batch and tenure to ensure it runs quickly but exercises the logic
        result_seq, result_energy = tabu_search_func(
            test_seq, 
            max_iterations=50, 
            tabu_tenure=5, 
            candidate_batch=16
        )
        
        # 4. Perform Assertions
        # Check return types
        assert isinstance(result_energy, (int, float, np.integer)), \
            f"FAIL: result_energy should be a number, got {type(result_energy)}"
        assert isinstance(result_seq, np.ndarray), \
            f"FAIL: result_seq should be a numpy array, got {type(result_seq)}"
        
        # Check logic: energy should never increase in a minimization search
        assert result_energy <= initial_energy, \
            f"FAIL: Energy increased! Initial: {initial_energy}, Final: {result_energy}"
        
        # Check output consistency
        recalculated_energy = int(to_cpu_func(energy_batch_func(xp_module.asarray([result_seq])))[0])
        assert recalculated_energy == result_energy, \
            f"FAIL: Reported energy ({result_energy}) does not match recalculated energy ({recalculated_energy})"

        print(f"  Initial energy: {initial_energy}")
        print(f"  Final energy:   {result_energy}")
        print("Tabu Search sanity check: PASS")
        
        # Reset seed to random
        np.random.seed(None)
        return True

    except Exception as e:
        print(f"Tabu Search check failed: {e}")
        np.random.seed(None)
        return False

### ENERGY ANALYSIS AND PLOTTING for GUIDED QUANTUM WALK
def run_energy_analysis_plot(sample_result, initial_bitstring, show_plot=True):
    """
    Analyzes the results of a Guided Quantum Walk and plots the energy distribution.
    Calculates initial energy, mean energy, and the best escape route.
    """
    def _calculate_labs_energy(bitstring):
        """Internal helper for sidelobe energy calculation."""
        # Handle both list of ints and string representations
        s = [1 if str(b) == '0' else -1 for b in bitstring]
        n = len(s)
        energy = 0
        for k in range(1, n):
            ck = sum(s[i] * s[i+k] for i in range(n - k))
            energy += ck**2
        return energy

    # 1. Energy Calculations
    initial_energy = _calculate_labs_energy(initial_bitstring)
    
    sample_energies = []
    for bitstring, count in sample_result.items():
        energy = _calculate_labs_energy(bitstring)
        sample_energies.extend([energy] * count)
    
    mean_energy = np.mean(sample_energies)
    min_energy = min(sample_energies)
    
    print(f"Energy Analysis Summary (N={len(initial_bitstring)}):")
    print(f"  - Initial Energy: {initial_energy:.2f}")
    print(f"  - Mean Sample Energy: {mean_energy:.2f}")
    print(f"  - Best Energy Found: {min_energy:.2f}")

    # 2. Setup Plot
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        # Distribution of Quantum Samples
        plt.hist(sample_energies, bins=20, color='skyblue', edgecolor='black', 
                 alpha=0.7, label='Quantum Walk Samples')
        
        # Reference Line: Initial Stuck State (Red)
        plt.axvline(initial_energy, color='red', linestyle='-', linewidth=2.5, 
                    label=f'Initial MTS Energy: {initial_energy:.2f}')
        
        # Reference Line: Mean of Samples (Orange)
        plt.axvline(mean_energy, color='darkorange', linestyle='--', linewidth=2.5, 
                    label=f'Mean Sample Energy: {mean_energy:.2f}')
        
        # Point: Best Found (Green)
        plt.scatter([min_energy], [0], color='green', s=100, zorder=5, 
                    label=f'Best "Escape" Route: {min_energy}')

        plt.title('LABS Guided Quantum Walk: Energy Distribution Analysis', fontsize=14)
        plt.xlabel('Sidelobe Energy (Lower is Better)', fontsize=12)
        plt.ylabel('Frequency (Shots)', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return min_energy, mean_energy


### MEMETIC TABU SEARCH VISUALIZATION AND STATS
def run_mts_visualization(best_seq, best_energy, population, energies, N):
    """
    Visualizes the final results of the Memetic Tabu Search.
    Displays energy distribution, sequence structure, and autocorrelation.
    """
    # Ensure inputs are standard numpy/python types for plotting
    if hasattr(best_seq, 'get'): best_seq = best_seq.get() # Handle CuPy
    if hasattr(energies, 'get'): energies = energies.get()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot 1: Energy distribution of final population ---
    axes[0].hist(energies, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(best_energy, color='red', linestyle='--', 
                    label=f'Best: {best_energy}')
    axes[0].set_xlabel('Energy $E$')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Energy Distribution (Final Pop)')
    axes[0].legend()

    # --- Plot 2: Best sequence visualization ---
    # Convert to 1/-1 if stored as 0/1 bits
    plot_seq = np.array([1 if x == 1 or x == 0 else -1 for x in best_seq])
    axes[1].bar(range(N), plot_seq, color=['#2ecc71' if s == 1 else '#e74c3c' for s in plot_seq])
    axes[1].set_xlabel('Index $i$')
    axes[1].set_ylabel('Spin $s_i$')
    axes[1].set_title(f'Best Sequence (N={N})')
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_yticks([-1, 1])
    
    # --- Plot 3: Autocorrelation of best sequence ---
    autocorr = []
    for k in range(1, N):
        C_k = sum(plot_seq[i] * plot_seq[i + k] for i in range(N - k))
        autocorr.append(C_k)
        
    axes[2].bar(range(1, N), autocorr, color='#3498db', alpha=0.8)
    axes[2].set_xlabel('Lag $k$')
    axes[2].set_ylabel('Autocorrelation $C_k$')
    axes[2].set_title('Sidelobes (Autocorrelation)')
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # --- Statistics Summary ---
    # Merit Factor is a common metric: MF = N^2 / (2 * Energy)
    merit_factor = (N**2) / (2 * best_energy) if best_energy != 0 else float('inf')
    
    print(f"\n{'='*60}")
    print(f"LABS OPTIMIZATION SUMMARY (N = {N})")
    print(f"{'='*60}")
    print(f"Best Energy found:  {best_energy}")
    print(f"Merit Factor (MF):  {merit_factor:.4f}")
    print(f"Mean Pop Energy:    {np.mean(energies):.2f} (±{np.std(energies):.2f})")
    print(f"Best Bitstring:     {''.join(['0' if s == 1 else '1' for s in plot_seq])}")
    print(f"{'='*60}\n")
    
    return merit_factor


### PLOTTING LABS ENERGY HISTOGRAM FROM QUANTUM SAMPLES
def plot_labs_histogram(result, filename='labs_histogram.png'):
    """
    Plots a weighted histogram of LABS energies from (score, count) results.
    """

    score_count_pairs = get_score_count_pairs(result)
    most_likely_score = compute_energy(bitstring_to_seq(result.most_probable()))
    
    # 1. Extract scores and counts
    scores = np.array([p[0] for p in score_count_pairs])
    counts = np.array([p[1] for p in score_count_pairs])

    # 2. Calculate Weighted Statistics
    total_shots = np.sum(counts)
    weighted_mean = np.sum(scores * counts) / total_shots
    min_energy = np.min(scores)

    # 3. Setup the Plot
    plt.figure(figsize=(12, 7))

    # Determine bin count manually (as 'auto' doesn't support weighted data)
    num_bins = min(int(np.max(scores) - np.min(scores) + 1), 50) if np.max(scores) > np.min(scores) else 1

    # Plot weighted histogram
    plt.hist(scores, weights=counts, bins=num_bins,
             color='royalblue', edgecolor='black', alpha=0.75)

    # 4. Add Vertical Markers
    plt.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
                label=f'Weighted Mean: {weighted_mean:.2f}')
    plt.axvline(min_energy, color='forestgreen', linestyle='-', linewidth=2,
                label=f'Min Energy (Best): {min_energy:.2f}')
    plt.axvline(most_likely_score, color='darkviolet', linestyle=':', linewidth=2,
                label=f'Most Likely Score: {most_likely_score:.2f}')

    # 5. Styling
    plt.title('LABS Energy Distribution (Quantum Samples)', fontsize=16)
    plt.xlabel('Energy (Lower is better)', fontsize=14)
    plt.ylabel('Total Frequency (Number of Shots)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")