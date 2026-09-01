"""Compare numpy reference outputs against the Rust kernel outputs for the
kern_check harness. Usage: python3 check_kern.py [dir]"""

import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/kern_check"

TOL = {
    "rms": 1e-5,
    "gemv": 1e-3,
    "gemm": 1e-3,
    "attn": 1e-3,
    "dn": 1e-3,
    "rope": 1e-5,
    "swiglu": 1e-5,
    "rph": 1e-5,
    "topk": 1e-3,
    "conv": 1e-3,
    "moe": 1e-3,
    "full_layer": 1e-3,
    "delta_layer": 1e-3,
}


def main():
    ok = True
    refs = sorted(f for f in os.listdir(OUT) if f.startswith("ref_") and f.endswith(".txt"))
    if not refs:
        print("no ref_*.txt found in", OUT)
        sys.exit(1)
    for ref_name in refs:
        rust_name = ref_name.replace("ref_", "rust_")
        ref = np.loadtxt(os.path.join(OUT, ref_name))
        rust_path = os.path.join(OUT, rust_name)
        if not os.path.exists(rust_path):
            print(f"MISSING {rust_name}")
            ok = False
            continue
        rust = np.loadtxt(rust_path)
        parts = ref_name.split("_")
        if parts[1] == "full" and parts[2] == "layer":
            kind = "full_layer"
            # number is parts[3] like "1.txt"
        elif parts[1] == "delta" and parts[2] == "layer":
            kind = "delta_layer"
            # number is parts[3] like "1.txt"
        else:
            kind = parts[1]
            # number is parts[2] like "1.txt"
        tol = TOL[kind]
        if ref.shape != rust.shape:
            print(f"{ref_name}: shape {rust.shape} != ref {ref.shape}")
            ok = False
            continue
        if kind == "rope":
            err = np.abs(rust - ref)
            worst = err.max()
            idx = int(err.argmax())
        else:
            denom = np.abs(ref) + 1e-6
            err = np.abs(rust - ref) / denom
            worst = err.max()
            idx = int(err.argmax())
        status = "OK " if worst <= tol else "FAIL"
        if worst > tol:
            ok = False
        print(f"{status} {ref_name}: n={ref.size} worst_rel={worst:.3e} "
              f"(at {idx}: rust={rust[idx]:.9e} ref={ref[idx]:.9e}) tol={tol:.0e}")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
