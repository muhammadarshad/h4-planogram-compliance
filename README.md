# H₄ Planogram Compliance

**A parameter-free, float-free pipeline for constrained retail vision.**

Planogram compliance — verifying that the right product occupies the right shelf
position — without floating point and without trained parameters, so it runs on
FPU-less, GPU-less edge hardware (ARM Cortex-M class) where convolutional and
frequency-domain methods cannot. Two halves:

- **Where (position):** the **H₄** transform encodes a bounding box `(x,y,w,h)` into a
  lossless 40-bit integer state in 8 add/subtracts; compliance is an integer L₁ threshold.
- **What (identity):** a **QRoPE** appearance descriptor retrieved (k-NN) against a mined
  product catalogue — also integer, also no trained weights.

## Results (real Grocery Dataset, measured)

| Task | Result | Baseline | Params |
|---|---|---|---|
| H₄ round-trip | **0 errors / 13,184 boxes** | — | 0 |
| Position compliance (AUC) | **0.987** | IoU 0.963 | 0 |
| Identity, top-1 (held-out shelves) | **79.7% ± 3.5** | colour 64.8%, CNN z.s. 88.8% | 0 |
| End-to-end violation detection | **100.0%** | — | 0 |
| End-to-end compliant confirmation | **72.4% ± 8.5** | — | 0 |

Identity uses a validation/test shelf split (config frozen on validation) and
leave-one-shelf-out k-NN; all numbers report mean ± 95% CI. Per-class accuracy is
uneven (88–98% on most classes, 8–28% on two minority classes) — see the paper's
limitations.

## Repository layout
```
paper/    h4_planogram_compliance.tex + figures
src/      validate_paper.py      T1–T4 + L1 separation + op count
          baseline_compare.py    position: H4 L1 vs IoU / L2
          retrieval_benchmark.py identity: QRoPE vs colour-hist vs CNN (leave-one-shelf-out)
          final_identity.py      identity with val/test split, per-class, CNN upper bound
          full_pipeline.py       end-to-end position + identity compliance
ports/    h4_core.c, h4_core.rs  H4 core in C and Rust (bit-exact with Python)
```

## Dataset (not included)
Uses the **Grocery Dataset** (Varol & Chaudhry), a separate repository.
Download it into `grocerydataset/` at the repo root:
```bash
git clone https://github.com/gulvarol/grocerydataset
# images: see that repo's release tarballs (ShelfImages, ProductImagesFromShelves)
```
Scripts expect `grocerydataset/annotations.csv`, `grocerydataset/ShelfImages/`,
and `grocerydataset/ProductImagesFromShelves/`.

## Reproduce
```bash
pip install -r requirements.txt
python3 src/validate_paper.py       # T1–T4, separation, op count
python3 src/baseline_compare.py     # position baselines (AUC table)
python3 src/retrieval_benchmark.py  # identity retrieval (needs torch for CNN row)
python3 src/final_identity.py       # val/test split + per-class + CNN upper bound
python3 src/full_pipeline.py        # end-to-end compliance
cc -O3 -o ports/h4 ports/h4_core.c && ./ports/h4   # C port (bit-exact)
```
CNN baselines require `torch`/`torchvision` (CPU is fine).

## Citation
See `CITATION.cff`. Author: Muhammad Arshad.

## License
CC BY-NC 4.0 (see `LICENSE`).
