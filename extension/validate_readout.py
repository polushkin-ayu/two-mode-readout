"""Independent validation: quadrature effects and all 256 outcome subsets.

No imports from check.py or readout.py. Run from any directory.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad

RESULTS = Path(__file__).resolve().parents[1]/'results'/'extension'
m = np.arange(-4, 5)
rho = (np.sqrt(5) - 1) / 2
masks = np.array(list(itertools.product([0, 1], repeat=8)))


def exact_effects():
    return np.array([[[
        quad(lambda x: np.cos(2*np.pi*(n-v)*x), b/8, (b+1)/8, epsabs=1e-12)[0]
        + 1j*quad(lambda x: np.sin(2*np.pi*(n-v)*x), b/8, (b+1)/8, epsabs=1e-12)[0]
        for n in m] for v in m] for b in range(8)])


def grid(q, shifts=1):
    out = np.zeros((8, 9, 9), complex)
    for j in range(q):
        for t in range(shifts):
            x = (j+(t+.5)/shifts)/q
            v = np.exp(2j*np.pi*m*x)
            out[int(x*8)] += np.outer(v.conj(), v)/(q*shifts)
    return out


def run():
    data = json.loads((RESULTS/'readout-results.json').read_text())
    exact = exact_effects()
    assert np.max(abs(exact.sum(axis=0)-np.eye(9))) < 1e-12
    for group in ['grids', 'shift_averaged']:
        for row in data[group]:
            q = row['Q']
            a = int(row['label'].split('/')[0])
            approx = grid(q, row['shifts'])
            curve = []
            for k in range(65):
                z = np.exp(2j*np.pi*m*k*(a/q-rho))
                delta = z.conj()[None,:,None]*approx*z[None,None,:]-exact
                curve.append(np.linalg.eigvalsh(np.einsum('sb,bij->sij', masks, delta)).max())
            discrepancy = np.max(abs(np.array(curve)-row['curve']))
            assert discrepancy < 1e-11
            print(row['label'], 'maximum TV', max(curve), 'curve discrepancy', discrepancy)
    for q in [16, 24, 144, 256]:
        diff = m[None,:]-m[:,None]
        ratio = np.ones((9,9))
        nonzero = diff != 0
        ratio[nonzero] = (np.pi*diff[nonzero]/q)/np.sin(np.pi*diff[nonzero]/q)
        assert np.max(abs(grid(q)-exact*ratio)) < 1e-12
    print('PASS: six grid curves, exact normalization, and aligned midpoint formula')


if __name__ == '__main__':
    run()
