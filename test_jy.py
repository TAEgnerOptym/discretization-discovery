from itertools import combinations, product

# ----------------------------
# Inputs: variable coverage and LP solution
# ----------------------------

var_to_elements = {
    'D': {'D'},
    'DC': {'D', 'C'},
    'A': {'A'},
    'BC': {'B', 'C'},
    'AB': {'A', 'B'},
    'ABC': {'A', 'B', 'C'},
}

lp_solution = {
    'D': 0.75,
    'DC': 0.25,
    'A': 0.25,
    'BC': 0.25,
    'AB': 0.25,
    'ABC': 0.5,
}

# ----------------------------
# Parameters
# ----------------------------

rhs_range = range(9)            # RHS values to test
coeff_choices = [0, 1, 2,3,4]       # Allowed LHS coefficients
max_subset_size = 7             # To control enumeration explosion
threshold = 1e-5                # For LP violation check

# ----------------------------
# Helper functions
# ----------------------------

def overlaps(v1, v2):
    return bool(v1 & v2)

def is_conflicting(subset, var_to_elements):
    """Returns True if any two variables in the subset overlap in coverage."""
    for i, vi in enumerate(subset):
        for vj in subset[i+1:]:
            if overlaps(var_to_elements[vi], var_to_elements[vj]):
                return True
    return False

def lp_value(subset, coeffs, lp_solution):
    return sum(coeffs[i] * lp_solution[subset[i]] for i in range(len(subset)))

def is_violated_by_any_integer_solution(subset, coeffs, rhs, var_to_elements, universe):
    """
    Checks if any 0/1 assignment to variables in 'subset' that fully covers
    'universe' violates the inequality defined by coeffs and rhs.
    """
    for assignment in product([0, 1], repeat=len(subset)):
        covered = set()
        for i, val in enumerate(assignment):
            if val == 1:
                covered.update(var_to_elements[subset[i]])
        relevant_universe = set().union(*(var_to_elements[v] for v in subset))
        if covered >= relevant_universe:
            lhs_val = sum(assignment[i] * coeffs[i] for i in range(len(subset)))
            if lhs_val > rhs:
                return True  # Violates the inequality
    return False  # Valid for all integer feasible solutions

# ----------------------------
# Main Search
# ----------------------------

universe = set().union(*var_to_elements.values())
variables = list(var_to_elements.keys())
violated_inequalities = []

for r in range(2, max_subset_size + 1):
    for subset in combinations(variables, r):
        if not is_conflicting(subset, var_to_elements):
            continue  # only consider overlapping variable sets

        for coeff_vector in product(coeff_choices, repeat=r):
            if all(c == 0 for c in coeff_vector):
                continue  # skip trivial all-zero inequalities

            lhs = lp_value(subset, coeff_vector, lp_solution)

            for rhs in rhs_range:
                if lhs > rhs + threshold:
                    # Check if this inequality is valid for all integer feasible solutions
                    if not is_violated_by_any_integer_solution(subset, coeff_vector, rhs, var_to_elements, universe):
                        violated_inequalities.append((subset, coeff_vector, rhs, lhs))

# ----------------------------
# Output
# ----------------------------

print(f"Found {len(violated_inequalities)} violated valid inequalities:\n")
print('violated_inequalities[0]')
print(violated_inequalities[0])
input('---')
for subset, coeffs, rhs, lhs_val in violated_inequalities:
    terms = [f"{a}·{v}" for a, v in zip(coeffs, subset) if a != 0]
    lhs_expr = " + ".join(terms)
    print(f"{lhs_expr} ≤ {rhs}   [LP value = {lhs_val:.4f}]")
