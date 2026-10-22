# Running the research implementation

The paper method uses five component states and a three-parameter Fingerprint reconstruction. The repository includes a runnable example on synthetic measurements; the paper's in-service monitoring records are not distributed here.

Install Python 3.10+ and dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the **single physics-guided model** using the paper-style offline preprocessing:

```bash
python -m aero_gas_path.demo --config configs/default.yaml --output-dir outputs/demo
```

For a short CPU smoke test:

```bash
python -m aero_gas_path.demo --epochs 2 --samples 96 --engines 2 --output-dir outputs/quickcheck
```

Optional development checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The generated evaluation files are written under `outputs/` and excluded by `.gitignore`. The decoder's 5×3 matrix uses the exact numerical entries from `fun_filter()` in the original supplied source script, normalized by column norms. Its bias is trainable. The encoder permits signed component states; the total loss is the paper's raw reconstruction sum of squares plus the sensitivity-normalized absolute-state penalty. Default preprocessing applies Romanovsky median/MAD screening and offline EEMD-NLM independently within each chronological partition.

**Numerical scope:** The noise-assisted EMD and NLM code is a compact implementation; its IMF significance handling and one-dimensional NLM do not fully reproduce every numerical detail of the published two-dimensional filtering algorithm. The code has not been validated on the paper's real in-service dataset. Offline preprocessing can use future observations within the processed partition and therefore is not suitable for strictly online predictive claims. Three gas-path observations do not uniquely determine five component states without additional information. Do not interpret synthetic errors as published experimental results.
