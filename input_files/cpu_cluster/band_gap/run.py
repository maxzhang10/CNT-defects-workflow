"""Compute and report the band gap of this replica via the Python API."""

import json
from pathlib import Path

from dpnegf.runner.cal_bandgap import BandGapCalculator

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "model.pth"
STRUCTURE = HERE / "structure.xyz"
RESULTS = HERE / "output/results"
NEL_ATOM = {"C": 4.0}  # Generated per replica from the XYZ and input template.

settings = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
task_options = settings["task_options"]
task_options.pop("task", None)
task_options["occupation_options"]["nel_atom"] = NEL_ATOM
task_options.setdefault("cache_options", {}).setdefault("path", str(RESULTS / "hs_blocks.npz"))

calculator = BandGapCalculator(
    structure=str(STRUCTURE),
    checkpoint=str(CHECKPOINT),
    dtype=settings["dtype"],
    task_options=task_options,
    results_path=str(RESULTS),
)
result = calculator.compute()

print(f"gap_status       : {result['gap_status']}")
print(f"signed gap       : {result['signed_gap_eV']:.4f} eV (nonnegative {result['band_gap_eV']:.4f} eV)")
print(f"min direct gap   : {result['minimum_direct_gap_eV']:.4f} eV")
print(f"VBM              : {result['vbm_eV']:.4f} eV at k={result['vbm_k_fractional']}")
print(f"CBM              : {result['cbm_eV']:.4f} eV at k={result['cbm_k_fractional']}")
print(f"converged        : {result['converged']} (tol {result['requested_resolution_eV']} eV)")
print(f"solver           : {result['solver']}")
print(f"H/S cache hit    : {result['cache']['hit']} "
      f"(spectrum hits {result['cache']['spectrum_hits']}, misses {result['cache']['spectrum_misses']})")
print("k convergence    :")
for level in result["k_convergence"]:
    change = level["max_edge_change_eV"]
    print(f"  nk={level['nk']:>3}  VBM={level['vbm_eV']: .4f}  CBM={level['cbm_eV']: .4f}"
          f"  dE={change if change is not None else float('nan'): .4f}  converged={level['converged']}")
