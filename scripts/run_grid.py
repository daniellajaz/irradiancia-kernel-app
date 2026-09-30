"""Ejecuta la rejilla completa (1 296 pipelines) fuera del notebook.
Uso (desde la raíz):  python scripts/run_grid.py
Es reanudable: si se interrumpe, al relanzarlo continúa donde quedó."""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from fkernel_lab.experiment import DATASETS, all_specs, load_dataset, run_grid  # noqa: E402

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    frames = {n: load_dataset(n) for n in DATASETS}
    run_grid(frames, all_specs(), "results/grid_results.csv")
    print("Rejilla completa")
