# two-mode-readout

Code for the teaching paper

> A. Yu. Polushkin, *Same evolution, different probabilities: a two-mode lesson
> on grid-readout error in quantum simulation* (manuscript for European Journal of Physics).

A wavefunction on a circle is restricted to two Fourier modes and read out by a
detector that reports which half of the circle contains the particle. The
rotation is simulated on a grid of `Q` nodes. The scripts compute, exactly, the
largest error in the predicted detector probability over all input states, and
show that this readout error can differ between grids that implement the same
rotation, and can outweigh the error of the rotation itself.

## Quick start

Python 3.9-3.12; `requirements.txt` pins the versions used.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python teaching.py
```

`teaching.py` runs in a few seconds and writes everything to `results/`.

## What the scripts produce

| Script | Output in `results/` | Used in the paper |
|---|---|---|
| `teaching.py` | `two-mode-results.json`: key `rows` | Table 1, rows 1-5 (comparisons for the golden-ratio rotation) |
| | `two-mode-results.json`: key `register_rows` | Table 1, last two rows: three-qubit register, nodes `j/8` versus midpoints (section 4.1) |
| | `parameter-sweep.csv` | 144 combinations of rotation, node count, shifts and horizon (teaching sequence, step 6) |
| | `figures/nodes`, `fringes`, `errors`, `mixture` (`.pdf`, `.png`) | Figures 1-4 |
| | `figures/convergence.pdf`, `.png` | Figure in the instructor notes |
| `extension/readout.py` | `extension/readout-results.json`, `extension/readout.{pdf,png}` | Appendix A: modes `|m| <= 4`, eight detector sectors, `k <= 64` |
| `extension/validate_readout.py` | (prints a report) | Independent check of the appendix curves |

The paper's supplementary material is in `docs/`: `student-worksheet.pdf` and
`instructor-notes.pdf`, which contains worked solutions.

Before writing results, `teaching.py` checks the closed-form error against
direct node sums, independently integrated detector effects, eigenvalues and
maximising states for `2 <= Q <= 64`, several shift counts, three rotation
angles and four times, as well as the formula for nodes at `j/Q`. It stops with
an assertion error if any check fails.

The appendix benchmark is optional:

```sh
python extension/readout.py
python extension/validate_readout.py
```

`readout.py` also contains a control comparison with rounded spectral phase
gates; it is retained for completeness and is not a quantitative claim of the
paper.

## Using it in class

The paper's teaching sequence asks students to predict, derive and then check
numbers. A natural computational exercise is to change the list of grids in
`run()` in `teaching.py`, for example other rotation angles, node counts or node
offsets `xi`, and to explain each change in ranking before running the code.
The function `error(q, a, rho, k, shifts)` implements the closed-form result for
midpoint grids; `error_offset(q, a, rho, k, xi)` computes the same quantity for
any node offset directly from eigenvalues.

## Numerical scope

All results are floating-point computations, not interval-certified proofs.
Results were generated with NumPy 2.0.2, SciPy 1.13.1 and Matplotlib 3.9.4,
and reproduced with NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2. Between
versions, printed errors agree to about `1e-15`; the stored maximising state
vectors may differ by a phase or, in degenerate cases, by the choice of
eigenvector.

## Licence and citation

MIT licence, see `LICENSE`. Citation metadata are in `CITATION.cff`.
