'''
Standard topological entanglement entropy values for common (2+1)D topological orders, organized by total quantum dimension D and gamma = ln(D).

!All unitary modular tensor categories with  D^2 <= 3 are pointed (Abelian).
'''


import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

# ----------------------------------------

@dataclass
class TopologicalOrderFamily:
    """
    Family of (2+1)D topological orders with a given total quantum dimension D.

    D^2 = sum_a d_a^2,  gamma = ln(D).
    This does NOT uniquely fix the modular tensor category, only the global invariant D.
    """
    name                     : str
    D                        : float
    gamma                    : float
    representative_examples  : List[Dict[str, Any]]
    description              : str

    def pretty(self) -> str:
        lines = []
        lines.append(f"name        : {self.name}")
        lines.append(f"D           : {self.D:.12g}")
        lines.append(f"gamma       : {self.gamma:.12g}")
        lines.append("examples     :")
        for ex in self.representative_examples:
            lines.append(f"  - {ex['name']} (abelian={ex['abelian']})")
            if ex.get("anyon_types") is not None:
                lines.append(f"    anyons : {ex['anyon_types']}")
            lines.append(f"    desc   : {ex['description']}")
        lines.append("description : " + self.description)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Global relations: gamma <-> D, and D^2 = sum_a d_a^2
    # ------------------------------------------------------------------

    @staticmethod
    def gamma_from_D(D: float) -> float:
        """
        Compute topological entanglement entropy gamma from total
        quantum dimension D:
            gamma = ln(D).
        """
        return float(np.log(D))

    @staticmethod
    def D_from_gamma(gamma: float) -> float:
        """
        Compute total quantum dimension D from gamma:
            D = exp(gamma).
        """
        return float(np.exp(gamma))

    def explain_global_relations(self) -> str:
        """
        Text explaining how D and gamma are related and what they constrain.

        - D is defined by:
              D^2 = sum_a d_a^2

          where d_a are the individual quantum dimensions of simple anyons.

        - The (universal) topological entanglement entropy is:
              gamma = ln(D).

        - Knowing only D (or gamma) fixes the global invariant D^2 but does
          not uniquely fix the set {d_a} or the full modular data (S, T).
        """
        s = []
        s.append("Global relations:")
        s.append("  D^2 = sum_a d_a^2, where d_a are quantum dimensions.")
        s.append("  gamma = ln(D).")
        s.append("Given only D (or gamma), many distinct modular tensor")
        s.append("categories can share the same global quantum dimension.")
        return "\n".join(s)

    # ------------------------------------------------------------------
    # Extraction of gamma from ED (Kitaev-Preskill / Levin-Wen)
    # ------------------------------------------------------------------

    @staticmethod
    def explain_ED_extraction() -> str:
        """
        Explain how gamma is extracted from ED / entanglement entropies.

        Standard setup:
          - Consider a gapped 2D topological phase on a large system.
          - For a simply connected region A with boundary length L:

                S(A) = alpha * L - gamma + o(1)

            where alpha is non-universal and gamma = ln(D) is the
            universal constant term.

        Practically, to eliminate alpha and geometry-dependent terms,
        one uses either:

          (1) Kitaev-Preskill construction:
                S_topo = S(A) + S(B) + S(C)
                         - S(AB) - S(BC) - S(CA)
                         + S(ABC)
                S_topo = -gamma

          (2) Levin-Wen construction:
                S_topo = combination of entropies for annular regions
                S_topo = -gamma

        In ED:
          - Choose regions A, B, C such that all linear sizes and
            separations are >> correlation length xi.
          - Compute von Neumann entropies S(X) for all required regions.
          - Form S_topo by the formula above.
          - Extract gamma as:
                gamma = -S_topo

        Finite-size corrections are O(e^{-L/xi}) plus geometry-dependent
        terms; for small tori these can be sizeable.
        """
        s  = []
        s.append("Extraction of gamma from ED:")
        s.append("  S(A) = alpha * L - gamma + o(1) for a simply connected A.")
        s.append("  Use a KP or Levin-Wen combination of S(X) to cancel alpha.")
        s.append("  For Kitaev-Preskill:")
        s.append("    S_topo = S(A) + S(B) + S(C)")
        s.append("             - S(AB) - S(BC) - S(CA)")
        s.append("             + S(ABC)")
        s.append("    gamma = -S_topo.")
        s.append("  All region sizes and separations must be >> correlation")
        s.append("  length to suppress finite-size and corner corrections.")
        return "\n".join(s)

    # ------------------------------------------------------------------
    # MES and access to individual anyon sectors
    # ------------------------------------------------------------------

    @staticmethod
    def explain_MES_usage() -> str:
        """
        Explain why MES are needed to access individual anyon quantum
        dimensions and statistics, given only ground states from ED.

        Setup:
          - Consider the system on a torus. A topological phase with
            N anyon types has N ground states |psi_i> in the thermodynamic
            limit.

          - In general, the ED eigenbasis |psi_i> is some arbitrary
            unitary mixing of the topological basis |a>, where a labels
            anyon flux threading a non-contractible cycle.

          - To access individual anyon quantum dimensions d_a and modular
            matrices (S, T), one constructs Minimally Entangled States (MES).

        MES idea:
          - For a fixed bipartition (e.g. cut along x-direction), consider
            all ground-state superpositions:

                |phi> = sum_i c_i |psi_i>.

          - Define the entanglement entropy S[phi] for that bipartition.
            MES are the states that minimize S[phi] subject to normalization.

          - It can be shown that, for a cut intersecting a given cycle,
            the MES basis aligns with eigenstates of a Wilson loop operator
            threading the orthogonal cycle. These MES are in one-to-one
            correspondence with anyon types a:

                |Xi_a>  <->  anyon a.

        Entanglement constants for MES:
          - For a MES |Xi_a> associated with anyon a, the entanglement
            entropy scales as:

                S_a(L) = alpha * L - ln(D) + ln(d_a) + o(1)
                       = alpha * L - ln(D / d_a) + o(1)

            where:
              D   = total quantum dimension,
              d_a = quantum dimension of anyon a.

          - Differences between MES entropies encode d_a:

                S_a(L) - S_1(L) = ln(d_a),

            where 1 is the vacuum sector (d_1 = 1).

        Practical ED usage:
          - Compute the low-energy ground-state manifold on a torus.
          - For a fixed cut, parameterize a general superposition of
            ground states and numerically minimize S across that cut
            to obtain N orthogonal MES |Xi_a>.
          - Use:
                S_a(L) - S_1(L) = ln(d_a)
            to extract individual quantum dimensions.
          - By using MES for different cuts (e.g. x and y cycles) and
            overlaps between MES bases, one reconstructs the modular
            S and T matrices.

        Summary:
          - gamma = ln(D) is a global invariant extracted from suitable
            combinations of entropies.
          - To resolve the individual anyon sectors (d_a, S, T), one
            must work in the MES basis, which requires:
                (i) access to all ground states on a torus, and
                (ii) entanglement-based minimization to construct MES.
        """
        s  = []
        s.append("MES usage for individual anyon data:")
        s.append("  On a torus with N anyon types, there are N ground states.")
        s.append("  ED gives some arbitrary basis {|psi_i>}; this is a unitary")
        s.append("  mixture of topological sectors |a> (anyon flux sectors).")
        s.append("  To identify individual anyons, construct Minimally")
        s.append("  Entangled States (MES) with respect to a fixed cut.")
        s.append("")
        s.append("  For a general ground-state superposition |phi> = sum_i c_i|psi_i>,")
        s.append("  minimize the entanglement entropy S[phi] across the cut.")
        s.append("  The resulting MES |Xi_a> are in one-to-one correspondence")
        s.append("  with anyon types a for that cut.")
        s.append("")
        s.append("  For a MES associated with anyon a:")
        s.append("    S_a(L) = alpha * L - ln(D) + ln(d_a) + o(1)")
        s.append("           = alpha * L - ln(D / d_a) + o(1).")
        s.append("  Hence:")
        s.append("    S_a(L) - S_1(L) = ln(d_a),  where 1 is the vacuum sector.")
        s.append("")
        s.append("  This allows extraction of individual quantum dimensions d_a")
        s.append("  from ED data. By constructing MES for cuts along different")
        s.append("  cycles and studying overlaps, one can also reconstruct the")
        s.append("  modular S and T matrices.")
        return "\n".join(s)


# ----------------------------------------

def su2_total_quantum_dimension(k: int) -> float:
    """
    Total quantum dimension of SU(2)_k:

        D_k^2 = (k + 2) / (2 sin^2(pi / (k + 2))).
    """
    x   = np.pi / (k + 2.0)
    D2  = (k + 2.0) / (2.0 * np.sin(x)**2)
    return float(np.sqrt(D2))

# ----------------------------------------

phi             = 0.5 * (1.0 + np.sqrt(5.0))        # golden ratio, appears in Fibonacci TQFT
D_fib           = float(np.sqrt(phi + 2.0))         # Fibonacci: d_1 = 1, d_tau = phi, so D^2 = 1 + phi^2 = phi + 2
gamma0          = 0.0
D_s2            = float(np.sqrt(2.0))               # D^2 = 2, e.g. semion theory
D_2             = 2.0
D_3             = 3.0
D_4             = 4.0

gamma_s2        = float(np.log(D_s2))
gamma_2         = float(np.log(D_2))
gamma_fib       = float(np.log(D_fib))
gamma_3         = float(np.log(D_3))
gamma_4         = float(np.log(D_4))

D_su2_3         = su2_total_quantum_dimension(3)
D_su2_4         = su2_total_quantum_dimension(4)
D_su2_5         = su2_total_quantum_dimension(5)
D_su2_6         = su2_total_quantum_dimension(6)

gamma_su2_3     = float(np.log(D_su2_3))
gamma_su2_4     = float(np.log(D_su2_4))
gamma_su2_5     = float(np.log(D_su2_5))
gamma_su2_6     = float(np.log(D_su2_6))

# ----------------------------------

TOPOLOGICAL_ENTROPIES: Dict[float, TopologicalOrderFamily] = {

    # D = 1  --------------------------------------------------------------
    gamma0 : TopologicalOrderFamily(
        name        = "D = 1 family",
        D           = 1.0,
        gamma       = gamma0,
        representative_examples = [
            {
                "name"        : "Trivial phase",
                "abelian"     : True,
                "anyon_types" : ["1"],
                "description" : "Short-range entangled; unique vacuum sector; no intrinsic topological order."
            }
        ],
        description = (
            "All theories with total quantum dimension D = 1 have only the vacuum sector with d_1 = 1. "
            "No anyonic excitations, no ground-state degeneracy on higher-genus manifolds. "
            "Stacking with SPT phases does not change D."
        ),
    ),

    # D = sqrt(2)
    gamma_s2 : TopologicalOrderFamily(
        name                    = "D = sqrt(2) family",
        D                       = D_s2,
        gamma                   = gamma_s2,
        representative_examples = [
            {
                "name"        : "U(1)_2 (semion) theory",
                "abelian"     : True,
                "anyon_types" : ["1", "s"],
                "description" : (
                    "Two Abelian anyons with d_1 = d_s = 1. "
                    "Topological spin theta_s = i (semion). "
                    "D^2 = 1^2 + 1^2 = 2."
                ),
            },
            {
                "name"        : "U(1)_{-2} (anti-semion) theory",
                "abelian"     : True,
                "anyon_types" : ["1", "s̄"],
                "description" : (
                    "Same fusion as U(1)_2, opposite chirality. "
                    "All anyons Abelian; D^2 = 2."
                ),
            },
        ],
        description = (
            "Here D^2 = 2. Any unitary modular tensor category with D^2 = 2 is pointed (all d_a = 1), "
            "so only Abelian topological orders are possible. The standard representative is the semion theory U(1)_2 "
            "and its conjugate. No non-Abelian theory can have D^2 = 2 (a non-Abelian sector requires d_a > 1, and then "
            "sum_a d_a^2 > 2)."
        ),
    ),

    # D = 2  --------------------------------------------------------------
    gamma_2 : TopologicalOrderFamily(
        name        = "D = 2 family",
        D           = D_2,
        gamma       = gamma_2,
        representative_examples = [
            {
                "name"        : "Z2 gauge theory / Toric code",
                "abelian"     : True,
                "anyon_types" : ["1", "e", "m", "ε"],
                "description" : (
                    "Four Abelian anyons, all with quantum dimension 1. "
                    "e and m are bosons with mutual semion statistics; ε = e × m is a fermion. "
                    "D^2 = 4 (four anyons)."
                ),
            },
            {
                "name"        : "Double-semion theory",
                "abelian"     : True,
                "anyon_types" : ["1", "s", "s̄", "b"],
                "description" : (
                    "Four Abelian anyons with d_a = 1. s and s̄ are semion and anti-semion; "
                    "b = s × s̄ is a boson. D^2 = 4. "
                    "Same global quantum dimension as the toric code but different braiding data."
                ),
            },
            {
                "name"        : "Three-fermion Z2 × Z2 topological order",
                "abelian"     : True,
                "anyon_types" : ["1", "f1", "f2", "f3"],
                "description" : (
                    "Four Abelian anyons with d_a = 1; f1, f2, f3 are all fermions with mutual semion statistics. "
                    "D^2 = 4. Distinct from toric code and double semion, but same D."
                ),
            },
            {
                "name"        : "Ising TQFT (SU(2)_2)",
                "abelian"     : False,
                "anyon_types" : ["1", "ψ", "σ"],
                "description" : (
                    "Non-Abelian theory with d_1 = 1, d_ψ = 1, d_σ = sqrt(2). "
                    "Fusion rules: σ × σ = 1 + ψ, σ × ψ = σ, ψ × ψ = 1. "
                    "D^2 = 1^2 + 1^2 + (sqrt(2))^2 = 4, so D = 2; same global dimension as toric code."
                ),
            },
        ],
        description = (
            "Here D^2 = 4. This allows both Abelian and non-Abelian topological orders. "
            "Abelian options include pointed theories with 4 anyons (Z2 gauge theory / toric code, double-semion, "
            "three-fermion model), all with d_a = 1. A non-Abelian example with the same D is the Ising TQFT (SU(2)_2), "
            "where one anyon has d = sqrt(2). Knowing only D = 2 (or gamma = ln 2) does not distinguish these phases; "
            "one needs the full modular data (S, T) or at least the individual quantum dimensions and spins."
        ),
    ),

    # Fibonacci-like global dimension  -----------------------------------
    gamma_fib : TopologicalOrderFamily(
        name        = "Fibonacci family (D^2 = phi + 2)",
        D           = D_fib,
        gamma       = gamma_fib,
        representative_examples = [
            {
                "name"        : "Chiral Fibonacci TQFT",
                "abelian"     : False,
                "anyon_types" : ["1", "τ"],
                "description" : (
                    "Non-Abelian theory with d_1 = 1, d_τ = phi = (1 + sqrt(5)) / 2. "
                    "Fusion rule: τ × τ = 1 + τ. "
                    "Total quantum dimension D^2 = 1 + phi^2 = phi + 2 ≈ 3.618, so D ≈ 1.902."
                ),
            },
            {
                "name"        : "Anti-Fibonacci (conjugate chiral partner)",
                "abelian"     : False,
                "anyon_types" : ["1", "τ̄"],
                "description" : (
                    "Same fusion category as Fibonacci, but opposite chirality (complex-conjugate topological spins). "
                    "Same global dimension D."
                ),
            },
        ],
        description = (
            "Here D^2 = phi + 2 with phi = (1 + sqrt(5)) / 2. This cannot be realized by a purely Abelian theory, "
            "since Abelian theories have D^2 equal to an integer (the number of simple objects). Therefore any theory "
            "with this D is intrinsically non-Abelian. The canonical examples are the chiral Fibonacci and its conjugate."
        ),
    ),

    # D = 3  --------------------------------------------------------------
    gamma_3 : TopologicalOrderFamily(
        name        = "D = 3 family",
        D           = D_3,
        gamma       = gamma_3,
        representative_examples = [
            {
                "name"        : "Z3 gauge theory / D(Z3)",
                "abelian"     : True,
                "anyon_types" : None,
                "description" : (
                    "Quantum double of Z3. There are 9 Abelian anyons, all with d_a = 1. "
                    "D^2 = 9, so D = 3. Fusion is that of Z3 × Z3̂ (charge × flux), with nontrivial braiding but all "
                    "anyons Abelian."
                ),
            },
        ],
        description = (
            "Here D^2 = 9. Any Abelian modular tensor category with 9 simple objects (all d_a = 1) realizes this D. "
            "A standard representative is the Z3 gauge theory D(Z3). For such small D, known classifications show that "
            "all unitary modular categories with D^2 = 9 are pointed (Abelian). Thus D = 3 implies Abelian topological "
            "order in this regime."
        ),
    ),

    # D = 4  --------------------------------------------------------------
    gamma_4 : TopologicalOrderFamily(
        name        = "D = 4 family",
        D           = D_4,
        gamma       = gamma_4,
        representative_examples = [
            {
                "name"        : "Z4 gauge theory / D(Z4)",
                "abelian"     : True,
                "anyon_types" : None,
                "description" : (
                    "Quantum double of Z4. There are 16 Abelian anyons, all with d_a = 1. "
                    "D^2 = 16, so D = 4. Fusion is that of Z4 × Z4̂ (charges and fluxes)."
                ),
            },
            {
                "name"        : "Z2 × Z2 gauge theory / D(Z2 × Z2)",
                "abelian"     : True,
                "anyon_types" : None,
                "description" : (
                    "Quantum double of Z2 × Z2, also with 16 Abelian anyons and D^2 = 16. "
                    "Different braiding data from Z4 gauge theory but the same global dimension."
                ),
            },
            {
                "name"        : "Ising × Ising",
                "abelian"     : False,
                "anyon_types" : None,
                "description" : (
                    "Product of two Ising TQFTs. Each Ising factor has D = 2, so D_total = 2 × 2 = 4. "
                    "Contains multiple non-Abelian sectors (σ ⊗ 1, 1 ⊗ σ, σ ⊗ σ, etc.)."
                ),
            },
        ],
        description = (
            "Here D^2 = 16. There are many Abelian and non-Abelian possibilities. Abelian examples include quantum doubles "
            "of groups of order 4 (Z4 and Z2 × Z2), each giving 16 Abelian anyons. Non-Abelian examples include product "
            "theories like Ising × Ising, Fibonacci × Fibonaccī, etc., which all share the same total quantum dimension D = 4. "
            "Global D alone is therefore far from sufficient to characterize the phase."
        ),
    ),

    # SU(2)_3  ------------------------------------------------------------
    gamma_su2_3 : TopologicalOrderFamily(
        name        = "SU(2)_3 family",
        D           = D_su2_3,
        gamma       = gamma_su2_3,
        representative_examples = [
            {
                "name"        : "SU(2)_3 Chern-Simons TQFT",
                "abelian"     : False,
                "anyon_types" : ["j = 0, 1/2, 1, 3/2"],
                "description" : (
                    "Non-Abelian theory with 4 primary fields. Quantum dimensions are "
                    "d_0 = 1, d_{3/2} = 1, and two non-Abelian fields with d > 1. "
                    "Total quantum dimension satisfies D^2 = 5 + sqrt(5)."
                ),
            },
        ],
        description = (
            "This entry is keyed by the global D of SU(2)_3 (a non-Abelian theory with D^2 = 5 + sqrt(5)). "
            "Other modular categories with the same D could in principle exist, but SU(2)_3 is the standard example. "
            "This D is not equal to 3 and not equal to phi^2; the often-quoted Fibonacci/doubled Fibonacci identifications "
            "must be treated carefully at the level of global dimension."
        ),
    ),

    # SU(2)_4  ------------------------------------------------------------
    gamma_su2_4 : TopologicalOrderFamily(
        name        = "SU(2)_4 family",
        D           = D_su2_4,
        gamma       = gamma_su2_4,
        representative_examples = [
            {
                "name"        : "SU(2)_4 Chern-Simons TQFT",
                "abelian"     : False,
                "anyon_types" : ["j = 0, 1/2, 1, 3/2, 2"],
                "description" : (
                    "5 primary fields. Quantum dimensions: d_0 = 1, d_2 = 1, d_1 = 2, "
                    "and two fields with d = sqrt(3). "
                    "D^2 = 12, so D = 2 * sqrt(3) ≈ 3.464."
                ),
            },
        ],
        description = (
            "Global quantum dimension corresponding to the SU(2)_4 Chern-Simons theory, a non-Abelian topological order. "
            "This D is sometimes misquoted as 6; the correct value is D^2 = 12, so D = 2 sqrt(3). "
            "Again, global D does not single out SU(2)_4 uniquely but it is the canonical example."
        ),
    ),

    # SU(2)_5  ------------------------------------------------------------
    gamma_su2_5 : TopologicalOrderFamily(
        name        = "SU(2)_5 family",
        D           = D_su2_5,
        gamma       = gamma_su2_5,
        representative_examples = [
            {
                "name"        : "SU(2)_5 Chern-Simons TQFT",
                "abelian"     : False,
                "anyon_types" : ["j = 0, 1/2, 1, 3/2, 2, 5/2"],
                "description" : (
                    "6 primary fields with a mix of Abelian and non-Abelian sectors. "
                    "Quantum dimensions follow the standard SU(2)_k formula; D^2 ≈ 18.59, D ≈ 4.312."
                ),
            },
        ],
        description = (
            "Non-Abelian SU(2)_5 theory. The total quantum dimension is given by the general SU(2)_k formula. "
            "The earlier heuristic identification D = 2 sqrt(6) is not correct for the SU(2)_5 global dimension; "
            "D is irrational and determined by D^2 = (k + 2) / (2 sin^2(pi/(k + 2)))."
        ),
    ),

    # SU(2)_6  ------------------------------------------------------------
    gamma_su2_6 : TopologicalOrderFamily(
        name        = "SU(2)_6 family",
        D           = D_su2_6,
        gamma       = gamma_su2_6,
        representative_examples = [
            {
                "name"        : "SU(2)_6 Chern-Simons TQFT",
                "abelian"     : False,
                "anyon_types" : ["j = 0, 1/2, 1, 3/2, 2, 5/2, 3"],
                "description" : (
                    "7 primary fields, mixture of Abelian and non-Abelian anyons. "
                    "D^2 = 16 + 8 sqrt(2), so D ≈ 5.226."
                ),
            },
        ],
        description = (
            "Non-Abelian SU(2)_6 theory with D^2 = 16 + 8 sqrt(2). "
            "Previously quoted simple values like D = 2 sqrt(10) are incorrect; the correct D follows from the SU(2)_k "
            "total quantum dimension formula. Other categories with this global D may exist, but SU(2)_6 is the standard example."
        ),
    ),
}

# ----------------------------------------
#! EOF
# ----------------------------------------