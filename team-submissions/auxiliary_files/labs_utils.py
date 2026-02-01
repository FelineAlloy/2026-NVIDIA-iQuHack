import cudaq
import numpy as np
import collections
from math import sin, cos, pi

# ==========================================
# 1. CLASSICAL ALGORITHMS (MTS & ENERGY)
# ==========================================

def compute_energy(s):
    """Compute LABS energy: E(s) = sum C_k^2"""
    N = len(s)
    energy = 0
    for k in range(1, N):
        C_k = sum(s[i] * s[i + k] for i in range(N - k))
        energy += C_k ** 2
    return energy

def initialize_population(pop_size, N):
    return [np.random.choice([-1, 1], size=N) for _ in range(pop_size)]

def combine(parent1, parent2):
    N = len(parent1)
    child = np.array([parent1[i] if np.random.random() < 0.5 else parent2[i] for i in range(N)])
    return child

def mutate(sequence, p_mutate_per_bit=None):
    N = len(sequence)
    if p_mutate_per_bit is None: p_mutate_per_bit = 1.0 / N
    mutated = sequence.copy()
    for i in range(N):
        if np.random.random() < p_mutate_per_bit: mutated[i] *= -1
    return mutated

def tabu_search(initial_seq, max_iterations=100, tabu_tenure=7):
    N = len(initial_seq)
    current = initial_seq.copy()
    best = current.copy()
    best_energy = compute_energy(best)
    tabu_list = [-tabu_tenure - 1] * N
    
    for iteration in range(max_iterations):
        best_neighbor = None
        best_neighbor_energy = float('inf')
        best_move = -1
        for i in range(N):
            is_tabu = (iteration - tabu_list[i]) < tabu_tenure
            neighbor = current.copy()
            neighbor[i] *= -1
            neighbor_energy = compute_energy(neighbor)
            if is_tabu and neighbor_energy >= best_energy: pass
            elif is_tabu: continue
            if neighbor_energy < best_neighbor_energy:
                best_neighbor = neighbor
                best_neighbor_energy = neighbor_energy
                best_move = i
        if best_neighbor is None: break
        current = best_neighbor
        tabu_list[best_move] = iteration
        if best_neighbor_energy < best_energy:
            best = best_neighbor.copy()
            best_energy = best_neighbor_energy
    return best, best_energy

def memetic_tabu_search(N, pop_size=20, generations=50, p_mutate=0.3,
                        tabu_iterations=50, tabu_tenure=7, 
                        initial_population=None, verbose=False):
    if initial_population is not None:
        population = [np.array(seq) for seq in initial_population]
    else:
        population = initialize_population(pop_size, N)
    
    energies = [compute_energy(seq) for seq in population]
    best_idx = np.argmin(energies)
    best_seq = population[best_idx].copy()
    best_energy = energies[best_idx]
    history = [best_energy]
    
    for gen in range(generations):
        idx1, idx2 = np.random.choice(pop_size, size=2, replace=False)
        child = combine(population[idx1], population[idx2])
        if np.random.random() < p_mutate: child = mutate(child)
        improved_child, child_energy = tabu_search(child, tabu_iterations, tabu_tenure)
        worst_idx = np.argmax(energies)
        if child_energy < energies[worst_idx]:
            population[worst_idx] = improved_child
            energies[worst_idx] = child_energy
        if child_energy < best_energy:
            best_seq = improved_child.copy()
            best_energy = child_energy
        history.append(best_energy)
    return best_seq, best_energy, population, energies, history

def get_classical_guess(N, timeout_seconds=None):
    best_seq, _, _, _, _ = memetic_tabu_search(N, pop_size=50, generations=20, verbose=False)
    return [1 if x == 1 else 0 for x in best_seq]

# ==========================================
# 2. PHASE 1: DCQO UTILS & KERNEL
# ==========================================
def get_interactions(N):
    G2, G4 = [], []
    for i in range(1, N - 1):
        for k in range(1, ((N - i) // 2) + 1):
            G2.append([i - 1, i + k - 1])
    for i in range(1, N - 2):
        limit_t = ((N - i - 1) // 2) + 1
        for t in range(1, limit_t):
            limit_k_start = t + 1
            limit_k_end = (N - i - t) + 1
            for k in range(limit_k_start, limit_k_end):
                G4.append([i - 1, i + t - 1, i + k - 1, i + k + t - 1])
    return G2, G4

# Helper for Phase 1 Kernel
@cudaq.kernel
def phase1_pauli_rot(reg: cudaq.qview, qubits: list[int], paulis: list[int], angle: float):
    n_qubits = len(qubits)
    for index in range(n_qubits):
        q = qubits[index]
        p = paulis[index]
        if p == 1: h(reg[q])
        elif p == 2: 
            sdg(reg[q])
            h(reg[q])
    last = qubits[-1]
    for q in qubits[:-1]: cx(reg[q], reg[last])
    rz(angle, reg[last])
    for index in range(n_qubits - 2, -1, -1):
        q = qubits[index]
        cx(reg[q], reg[last])
    for index in range(n_qubits - 1, -1, -1):
        q = qubits[index]
        p = paulis[index]
        if p == 1: h(reg[q])
        elif p == 2:
            h(reg[q])
            s(reg[q])

@cudaq.kernel
def create_dcqo_kernel(N: int, G2: list[list[int]], G4: list[list[int]], steps: int, dt: float, T: float, thetas: list[float]):
    reg = cudaq.qvector(N)
    h(reg) # Superposition
    for step in range(steps):
        theta = thetas[step]
        theta2 = 4.0 * theta
        for pair in G2:
            phase1_pauli_rot(reg, pair, [2,0], theta2)
            phase1_pauli_rot(reg, pair, [0,2], theta2)
        theta4 = 8.0 * theta
        for quad in G4:
            phase1_pauli_rot(reg, quad, [2,0,0,0], theta4)
            phase1_pauli_rot(reg, quad, [0,2,0,0], theta4)
            phase1_pauli_rot(reg, quad, [0,0,2,0], theta4)
            phase1_pauli_rot(reg, quad, [0,0,0,2], theta4)

# ==========================================
# 3. PHASE 2: WS-QAOA KERNEL
# ==========================================
def apply_pauli_string_rotation(kernel, q_reg, qubits_indices, pauli_encodings, angle):
    n_qubits = len(qubits_indices)
    if n_qubits == 0: return
    for i in range(n_qubits):
        q_idx = qubits_indices[i]
        p_type = pauli_encodings[i]
        if p_type == 1: kernel.h(q_reg[q_idx])
        elif p_type == 2:
            kernel.sdg(q_reg[q_idx])
            kernel.h(q_reg[q_idx])
    last_idx = qubits_indices[-1]
    for i in range(n_qubits - 1): kernel.cx(q_reg[qubits_indices[i]], q_reg[last_idx])
    kernel.rz(angle, q_reg[last_idx])
    for i in range(n_qubits - 2, -1, -1): kernel.cx(q_reg[qubits_indices[i]], q_reg[last_idx])
    for i in range(n_qubits - 1, -1, -1):
        q_idx = qubits_indices[i]
        p_type = pauli_encodings[i]
        if p_type == 1: kernel.h(q_reg[q_idx])
        elif p_type == 2:
            kernel.h(q_reg[q_idx])
            kernel.s(q_reg[q_idx])

def create_ws_qaoa_kernel(N, layers):
    kernel, thetas, params = cudaq.make_kernel(list[float], list[float])
    q = kernel.qalloc(N)
    for i in range(N): kernel.ry(thetas[i], q[i])
    
    labs_terms = []
    for k in range(1, N):
        limit = N - k
        for i in range(limit):
            for j in range(limit):
                raw = [i, i+k, j, j+k]
                cnt = collections.Counter(raw)
                active = sorted([idx for idx, c in cnt.items() if c % 2 == 1])
                if active: labs_terms.append(active)

    for layer in range(layers):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        rot_angle = 2.0 * gamma
        for term in labs_terms:
            apply_pauli_string_rotation(kernel, q, term, [0]*len(term), rot_angle)
        for i in range(N):
            kernel.ry(-1.0 * thetas[i], q[i])
            kernel.rz(-2.0 * beta, q[i])
            kernel.ry(thetas[i], q[i])
    return kernel

# ==========================================
# 4. MATH UTILS
# ==========================================
def compute_topology_overlaps(G2, G4):
    def count_matches(list_a, list_b):
        matches = 0
        set_b = set(tuple(sorted(x)) for x in list_b)
        for item in list_a:
            if tuple(sorted(item)) in set_b: matches += 1
        return matches
    return {'22': count_matches(G2, G2), '44': count_matches(G4, G4), '24': 0}

def compute_theta(t, dt, total_time, N, G2, G4):
    if total_time == 0: return 0.0
    arg = (pi * t) / (2.0 * total_time)
    lam = sin(arg)**2
    lam_dot = (pi / (2.0 * total_time)) * sin((pi * t) / total_time)
    term_g1_2 = 16 * len(G2) * 2
    term_g1_4 = 64 * len(G4) * 4
    Gamma1 = term_g1_2 + term_g1_4
    sum_G2 = len(G2) * (lam**2 * 2)
    sum_G4 = 4 * len(G4) * (16 * (lam**2) + 8 * ((1 - lam)**2))
    I_vals = compute_topology_overlaps(G2, G4)
    term_topology = 4 * (lam**2) * (4 * I_vals['24'] + I_vals['22']) + 64 * (lam**2) * I_vals['44']
    Gamma2 = -256 * (term_topology + sum_G2 + sum_G4)
    alpha = 0.0 if abs(Gamma2) < 1e-12 else - Gamma1 / Gamma2
    return dt * alpha * lam_dot