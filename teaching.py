"""Two Fourier modes and a binary detector: the teaching example of the paper.

Run from any directory. Uses NumPy, SciPy and Matplotlib. The table data,
a parameter sweep, Figures 1-4 of the paper and the convergence figure of the
instructor notes are written to results/ next to this script.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent/'results'
RHO = (np.sqrt(5)-1)/2
MODES = np.array([0, 1])
E = np.array([[.5, 1j/np.pi], [-1j/np.pi, .5]])


def grid(q, shift=.5):
    """Effect of [0,1/2); a boundary node belongs to the other outcome."""
    x = (np.arange(q)+shift)/q
    f = np.exp(2j*np.pi*np.outer(x, MODES))/np.sqrt(q)
    return f[x < .5].conj().T @ f[x < .5]


def parameters(q):
    if q % 2 == 0:
        return 0., 1j/(q*np.sin(np.pi/q))
    return -1/(2*q), (1+1j/np.tan(np.pi/(2*q)))/(2*q)


def error(q, a, rho, k, shifts=1):
    # S equispaced shifts reproduce Q*S nodes, with dynamics still a/Q.
    d, g = parameters(q*shifts)
    return abs(d)+abs(g*np.exp(2j*np.pi*k*(a/q-rho))-1j/np.pi)


def error_offset(q, a, rho, k, xi):
    """Worst-case error for one grid with node offset xi, by direct eigenvalues."""
    z = np.diag(np.exp(2j*np.pi*MODES*k*(a/q-rho)))
    return max(abs(np.linalg.eigvalsh(z.conj().T@grid(q,xi)@z-E)))


def validate():
    independent = np.array([[quad(lambda x: np.cos(2*np.pi*(n-m)*x),0,.5)[0]
                            +1j*quad(lambda x: np.sin(2*np.pi*(n-m)*x),0,.5)[0]
                            for n in MODES] for m in MODES])
    assert np.max(abs(independent-E)) < 1e-13
    residual = 0.
    for q in range(2,65):
        for s in [1,2,3,4]:
            effect = sum(grid(q,(r+.5)/s) for r in range(s))/s
            d,g = parameters(q*s)
            assert np.max(abs(effect-np.array([[.5+d,g],[g.conjugate(),.5+d]]))) < 1e-12
            for rho in [RHO,np.sqrt(2)-1,np.pi-3]:
                a = round(q*rho)
                for k in [0,1,8,32]:
                    z = np.diag(np.exp(2j*np.pi*MODES*k*(a/q-rho)))
                    delta = z.conj().T@effect@z-independent
                    values, vectors = np.linalg.eigh(delta)
                    direct = max(abs(values))
                    analytic = error(q,a,rho,k,s)
                    residual = max(residual,abs(direct-analytic))
                    c = vectors[:,np.argmax(abs(values))]
                    assert abs(abs(np.vdot(c,delta@c))-analytic) < 1e-12
    # Power-of-two style nodes j/Q (xi=0), even Q: d=0, g=(1+i cot(pi/Q))/Q.
    for q in range(2,65,2):
        g = (1+1j/np.tan(np.pi/q))/q
        assert np.max(abs(grid(q,0.)-np.array([[.5,g],[g.conjugate(),.5]]))) < 1e-12
    assert residual < 1e-12
    return residual


BLUE, ORANGE, PURPLE, INK = '#2563eb', '#c2410c', '#7c3aed', '#222222'
SIDE0, SIDE1 = '#dbeafe', '#ffedd5'


def d_g(q, xi):
    """Diagonal shift d and off-diagonal element g of the grid effect."""
    effect = grid(q, xi)
    return effect[0, 0].real-.5, effect[0, 1]


def fringe(d, g, phi):
    """Predicted p0 for the balanced input (1, exp(i phi))/sqrt(2)."""
    return .5+d+np.real(g*np.exp(1j*phi))


def save(fig, name):
    fig.savefig(OUT/f'figures/{name}.pdf')
    fig.savefig(OUT/f'figures/{name}.png', dpi=200)
    plt.close(fig)


def figure_nodes():
    """Figure 1: node sums for two prescribed inputs."""
    t = np.linspace(0, 1, 400)
    panels = [(5, .5, -1, r'(a) $Q=5$, midpoints'), (10, .5, -1, r'(b) $Q=10$, midpoints'),
              (8, 0., 1, r'(c) $Q=8$, nodes $j/8$'), (8, .5, 1, r'(d) $Q=8$, midpoints')]
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 3.9), layout='constrained', sharex=True, sharey=True)
    for ax, (q, xi, sign, title) in zip(axs.flat, panels):
        density = lambda x: 1+sign*np.cos(2*np.pi*x)
        ax.axvspan(0, .5, color=SIDE0, lw=0)
        ax.axvspan(.5, 1, color=SIDE1, lw=0)
        ax.plot(t, density(t), color=INK, lw=1.2)
        x = (np.arange(q)+xi)/q
        y, inside = density(x), x < .5
        colours = np.where(inside, BLUE, ORANGE)
        ax.vlines(x, 0, y, color=colours, lw=2)
        ax.scatter(x, y, c=colours, s=24, zorder=3, clip_on=False)
        boundary = np.isclose(x, .5) | np.isclose(x, 0.)
        if boundary.any():
            ax.scatter(x[boundary], y[boundary], s=90, facecolors='none', edgecolors='black',
                       lw=1, zorder=4, clip_on=False)
        predicted = y[inside].sum()/q
        assert abs(predicted-np.real(np.array([1, sign])@grid(q, xi)@np.array([1, sign]))/2) < 1e-12
        ax.text(.25, 2.2, f'nodes: {predicted:.3g}', ha='center', fontweight='bold')
        ax.text(.25, 1.97, 'exact: 0.5', ha='center', fontsize=8.5)
        ax.set(xlim=(-.04, 1.02), ylim=(0, 2.5), xticks=[0, .5, 1], yticks=[0, 1, 2])
        ax.set_title(title, fontsize=9, loc='left')
    for ax in axs[1]:
        ax.set_xlabel(r'Angle $\theta$ (turns)')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'$|\psi(\theta)|^2$')
    save(fig, 'nodes')


def figure_fringes():
    """Figure 2: the worst-case formula as a fringe gap and as a rotating phasor."""
    phi = np.linspace(0, 2*np.pi, 400)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.3, 4.6), layout='constrained', height_ratios=[1.25, 1])
    exact = .5-np.sin(phi)/np.pi
    ax.plot(phi/np.pi, exact, color=INK, lw=2.2, label='exact')
    for q, xi, colour, style, label in [(5, .5, ORANGE, '-', r'$Q=5$, midpoints'),
                                        (8, 0., PURPLE, '-', r'$Q=8$, nodes $j/8$'),
                                        (8, .5, PURPLE, '--', r'$Q=8$, midpoints')]:
        d, g = d_g(q, xi)
        curve = fringe(d, g, phi)
        # The largest gap over balanced inputs is the worst-case error at k=0.
        assert abs(np.max(abs(curve-exact))-error_offset(q, 1, RHO, 0, xi)) < 1e-4
        ax.plot(phi/np.pi, curve, color=colour, ls=style, lw=1.3, label=label)
    ax.set(xlabel=r'Relative phase $\phi/\pi$', ylabel=r'Predicted $p_0$', xlim=(0, 2), ylim=(.05, .9))
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.set_title(r'(a) Sampled fringes at $k=0$; the largest vertical gap is $\varepsilon_0$', fontsize=9, loc='left')
    t = np.linspace(np.pi/2-.6, np.pi/2+.6, 200)
    bx.plot(np.cos(t)/np.pi, np.sin(t)/np.pi, color='0.8', lw=1)
    bx.scatter([0], [1/np.pi], s=90, color=INK, zorder=7, marker='*')
    bx.text(0, .2965, r'exact $\mathrm{i}/\pi$', ha='center', fontsize=8.5)
    for q, a, xi, colour, label, x0, y0 in [(5, 3, .5, ORANGE, r'$3/5$, $Q=5$', .066, .2935),
                                            (6, 4, .5, BLUE, r'$4/6$, $Q=6$', .006, .3385),
                                            (8, 5, 0., PURPLE, r'$5/8$, nodes $j/8$', .092, .3195),
                                            (8, 5, .5, PURPLE, r'$5/8$, midpoints', -.078, .3315)]:
        d, g = d_g(q, xi)
        z = g*np.exp(2j*np.pi*np.arange(4)*(a/q-RHO))
        # |d| + distance to i/pi reproduces the closed form at every plotted k.
        for k in range(4):
            assert abs(abs(d)+abs(z[k]-1j/np.pi)-error_offset(q, a, RHO, k, xi)) < 1e-12
        bx.plot(z.real, z.imag, '-o', color=colour, ms=3.5, lw=1.1)
        bx.scatter(z.real[0], z.imag[0], s=36, facecolors='white', edgecolors=colour, zorder=6, lw=1.3)
        bx.text(x0, y0, label, fontsize=8.5, color=INK)
    bx.set(xlabel=r'Re $g$', ylabel=r'Im $g$', aspect='equal', xlim=(-.13, .17), ylim=(.285, .345))
    bx.set_title(r'(b) Evolution rotates $g$ about the origin: open circle $k=0$, then $k=1,2,3$', fontsize=9, loc='left')
    save(fig, 'fringes')


def figure_errors():
    """Figure 3: worst-case error against the number of translations."""
    ks = np.arange(9)
    fig, ax = plt.subplots(figsize=(4.8, 2.9), layout='constrained')
    for label, f, style, colour in [(r'$3/5$, $Q=5$ midpoints', lambda k: error(5, 3, RHO, k), 'o-', ORANGE),
                                    (r'$4/6$, $Q=6$ midpoints', lambda k: error(6, 4, RHO, k), 's-', BLUE),
                                    (r'$5/8$, nodes $j/8$', lambda k: error_offset(8, 5, RHO, k, 0.), '^-', PURPLE),
                                    (r'$5/8$, $Q=8$ midpoints', lambda k: error_offset(8, 5, RHO, k, .5), '^--', PURPLE),
                                    (r'$3/5$, two shifts', lambda k: error(5, 3, RHO, k, 2), 'o--', ORANGE)]:
        ax.plot(ks, [f(k) for k in ks], style, color=colour, ms=4, lw=1.3, label=label)
    ax.set(xlabel=r'Number of translations $k$', ylabel='Worst-case probability error', ylim=(0, .66))
    ax.legend(fontsize=7.5, frameon=False, loc='upper left', ncol=2)
    save(fig, 'errors')


def figure_mixture():
    """Figure 4: two shifted settings are each biased; their average is not."""
    phi = np.linspace(0, 2*np.pi, 400)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.7), layout='constrained', width_ratios=[1, 1.15])
    ax.axvspan(0, .5, color=SIDE0, lw=0)
    ax.axvspan(.5, 1, color=SIDE1, lw=0)
    for y, q, xi in [(2, 5, .25), (1, 5, .75), (0, 10, .5)]:
        x = (np.arange(q)+xi)/q
        ax.scatter(x, np.full(q, y), c=np.where(x < .5, BLUE, ORANGE), s=22, zorder=3)
        ax.text(.25, y+.2, f'{(x < .5).sum()} of {q} nodes', ha='center', fontsize=8)
    ax.set(xlim=(-.02, 1.02), ylim=(-.5, 2.7), yticks=[2, 1, 0], xlabel=r'Angle $\theta$ (turns)',
           yticklabels=[r'$\xi=1/4$', r'$\xi=3/4$', 'union'])
    ax.set_title(r'(a) Interlaced nodes, $Q=5$', fontsize=9, loc='left')
    exact = .5-np.sin(phi)/np.pi
    one, other = fringe(*d_g(5, .25), phi), fringe(*d_g(5, .75), phi)
    bx.plot(phi/np.pi, exact, color=INK, lw=2.4, label='exact')
    bx.plot(phi/np.pi, one, color=ORANGE, lw=1.2, label=r'$\xi=1/4$')
    bx.plot(phi/np.pi, other, color=BLUE, lw=1.2, label=r'$\xi=3/4$')
    bx.plot(phi/np.pi, (one+other)/2, color='#16a34a', ls='--', lw=1.5, label='average')
    bx.set(xlabel=r'Relative phase $\phi/\pi$', ylabel=r'Predicted $p_0$', xlim=(0, 2), ylim=(.05, 1.08))
    bx.legend(fontsize=8, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.03))
    bx.set_title('(b) Each setting is biased; the average is not', fontsize=9, loc='left')
    save(fig, 'mixture')


def figure_convergence():
    """Instructor notes: readout convergence at k=0."""
    qs = np.arange(5, 201)
    odd, even = qs[qs % 2 == 1], qs[qs % 2 == 0]
    fig, ax = plt.subplots(figsize=(4.6, 2.8), layout='constrained')
    ax.loglog(odd, [error_offset(q, 1, RHO, 0, .5) for q in odd], color=ORANGE, lw=1.5, label='odd $Q$, midpoints')
    ax.loglog(even, [error_offset(q, 1, RHO, 0, 0.) for q in even], color=PURPLE, ls='--', lw=1.5, label='even $Q$, nodes $j/Q$')
    ax.loglog(even, [error_offset(q, 1, RHO, 0, .5) for q in even], color=BLUE, lw=1.5, label='even $Q$, midpoints')
    ax.set(xlabel='Number of nodes $Q$', ylabel=r'$\varepsilon_0$')
    ax.legend(fontsize=8, frameon=False)
    save(fig, 'convergence')


def figures():
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure_nodes()
    figure_fringes()
    figure_errors()
    figure_mixture()
    figure_convergence()


def run():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    residual=validate()
    rows=[]
    for q,a,s in [(5,3,1),(10,6,1),(6,4,1),(5,3,2),(6,4,2)]:
        rows.append(dict(Q=q,a=a,shifts=s,slope_error=abs(a/q-RHO),
                         TV_k0=error(q,a,RHO,0,s),TV_k1=error(q,a,RHO,1,s)))
    register_rows=[]
    for q,a,xi in [(8,5,0.),(8,5,.5)]:
        register_rows.append(dict(Q=q,a=a,offset=xi,slope_error=abs(a/q-RHO),
                                  TV_k0=error_offset(q,a,RHO,0,xi),TV_k1=error_offset(q,a,RHO,1,xi)))
    (OUT/'two-mode-results.json').write_text(json.dumps({'rho':RHO,'validation_residual':residual,'rows':rows,
                                                         'register_rows':register_rows},indent=2)+'\n')
    with (OUT/'parameter-sweep.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['rho_name','rho','Q','a','shifts','K','slope_error','max_TV'])
        writer.writeheader()
        for name,rho in [('golden',RHO),('sqrt2_minus1',np.sqrt(2)-1),('pi_minus3',np.pi-3)]:
            for q in [5,6,8,13,16,21,32,34]:
                for s in [1,2]:
                    a=round(q*rho)
                    for K in [1,8,32]:
                        writer.writerow(dict(rho_name=name,rho=rho,Q=q,a=a,shifts=s,K=K,
                                             slope_error=abs(a/q-rho),max_TV=max(error(q,a,rho,k,s) for k in range(K+1))))
    figures()
    print(json.dumps(rows,indent=2));print(json.dumps(register_rows,indent=2))
    print('PASS: independent effects, grid sums, eigennorms, witness states and j/Q nodes; residual',residual)

if __name__=='__main__':run()
