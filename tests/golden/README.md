# Golden Reference Tests

Python scripts for cross-validating Rust kernels against numpy implementations.

## Usage

```bash
# 1. Generate synthetic inputs and numpy reference outputs
python3 tests/golden/mkkern.py /tmp/kern_data

# 2. Run Rust kernels and dump outputs
cargo run --bin kern_check /tmp/kern_data

# 3. Compare Rust outputs against numpy references
python3 tests/golden/check_kern.py /tmp/kern_data
```

## Files

- `mkkern.py` — Generates synthetic test inputs and numpy reference outputs
- `check_kern.py` — Compares `rust_*.txt` vs `ref_*.txt` with per-kernel tolerances
- `spec.txt` — Test specification: kernel type, dimensions, tolerances

## Adding Tests

Append a line to `spec.txt` with the format:
```
kernel_name test_id param1 param2 ... [tolerance]
```
Then add the corresponding Rust case in `src/bin/kern_check.rs`.
