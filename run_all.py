"""Run all required NumPy checks for Homework 1."""

import runpy


for script in (
    "q1_basic_operations.py",
    "q3_system.py",
    "q4_determinant_rank.py",
    "q5_inverse.py",
):
    print(f"\n{'=' * 16} {script} {'=' * 16}")
    runpy.run_path(script, run_name="__main__")
