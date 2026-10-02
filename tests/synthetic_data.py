# tests/synthetic_data.py
# Synthetic kinetic datasets for tests.  These values are computer-generated
# archetypes, NOT real experimental measurements, so they are safe to commit to
# a public repository.

# Sparse 5-point time grid in minutes (no t=0 measurement).
T_SPARSE = [20, 40, 80, 180, 240]

# Removal (%) per archetype, keyed by a short descriptive name.
SYNTHETIC_REMOVAL = {
    "A_first_order": [21.9, 39.5, 61.4, 91.3, 95.9],  # clean PFO
    "B_initial_drop": [35.7, 45.7, 61.8, 83.2, 89.8],  # ~24 % fast drop, then PFO
    "C_plateau": [21.8, 26.0, 34.3, 47.1, 53.1],  # slows early, levels off
    "D_dark_like": [7.9, 13.2, 19.1, 25.9, 28.1],  # slow adsorption control
}
