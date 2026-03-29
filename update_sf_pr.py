import re

with open("physics/response/structure_factor.py", "r") as f:
    content = f.read()

# Define constant and apply formatting fixes
content = content.replace("def structure_factor_spin(", "SPARSE_MATRIX_THRESHOLD = 1e-15\n\ndef structure_factor_spin(")

# T=0 block
old_t0 = """        # ⚡ Bolt: Optimization - Filter out zero matrix elements to avoid useless kernel iterations
        mask = matrix_elements > 1e-15
        energy_diffs = energy_diffs[mask]
        matrix_elements = matrix_elements[mask]"""

new_t0 = """        # Optimization: Filter out zero matrix elements to avoid useless kernel iterations
        mask = matrix_elements > SPARSE_MATRIX_THRESHOLD
        if not np.all(mask):
            energy_diffs = energy_diffs[mask]
            matrix_elements = matrix_elements[mask]"""

content = content.replace(old_t0, new_t0)


# T>0 block
old_t1 = """            # ⚡ Bolt: Optimization - Filter out zero matrix elements
            mask = matrix_elements > 1e-15
            if not np.any(mask):
                continue

            # Energy differences
            energy_diffs = eigvals[mask] - E_i
            matrix_elements = matrix_elements[mask]"""

new_t1 = """            # Optimization: Filter out zero matrix elements to avoid useless kernel iterations
            mask = matrix_elements > SPARSE_MATRIX_THRESHOLD
            if not np.any(mask):
                continue

            # Energy differences
            if not np.all(mask):
                energy_diffs = eigvals[mask] - E_i
                matrix_elements = matrix_elements[mask]
            else:
                energy_diffs = eigvals - E_i"""

content = content.replace(old_t1, new_t1)

with open("physics/response/structure_factor.py", "w") as f:
    f.write(content)
