"""Two Fourier modes and a binary detector: the teaching example of the paper.

Run from any directory. Uses NumPy, SciPy and Matplotlib. The table data,
the three-qubit register comparison, a parameter sweep and Figure 1 are
written to results/ next to this script.
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
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
    ax=axs[0]
    ax.axvspan(0,.5,color='#dbeafe');ax.axvspan(.5,1,color='#ffedd5')
    grids=[(5,5,.5,'Q=5, midpoint'),(4,6,.5,'Q=6, midpoint'),(3,8,0.,'Q=8, nodes j/8'),
           (2,8,.5,'Q=8, midpoint'),(1,5,.25,'Q=5, shift 1/4'),(0,5,.75,'Q=5, shift 3/4')]
    for y,q,shift,label in grids:
        x=(np.arange(q)+shift)/q
        ax.scatter(x,np.full(q,y),c=np.where(x<.5,'#2563eb','#c2410c'),s=35,zorder=3)
        onb=np.isclose(x,.5)|np.isclose(x,0.)
        if onb.any():
            ax.scatter(x[onb],np.full(onb.sum(),y),s=150,facecolors='none',edgecolors='black',lw=1.2,zorder=4)
    ax.axvline(.5,color='black',ls=':',lw=1);ax.axvline(0.,color='black',ls=':',lw=1)
    ax.axhline(3.5,color='0.6',lw=.6);ax.axhline(1.5,color='0.6',lw=.6)
    ax.set(xlim=(-.04,1.02),ylim=(-.5,5.9),yticks=[g[0] for g in grids],yticklabels=[g[3] for g in grids],
           xlabel='Angle (turns)',title='(a) Node placement relative to the detector')
    ax.text(.25,5.5,'outcome 0',ha='center',fontsize=9);ax.text(.75,5.5,'outcome 1',ha='center',fontsize=9)
    ax=axs[1];ks=np.arange(9)
    curves=[('3/5, Q=5 midpoint',lambda k:error(5,3,RHO,k),'o-','#c2410c'),
            ('4/6, Q=6 midpoint',lambda k:error(6,4,RHO,k),'s-','#2563eb'),
            ('5/8, nodes j/8',lambda k:error_offset(8,5,RHO,k,0.),'^-','#7c3aed'),
            ('5/8, Q=8 midpoint',lambda k:error_offset(8,5,RHO,k,.5),'^--','#7c3aed'),
            ('3/5, two shifts',lambda k:error(5,3,RHO,k,2),'o--','#c2410c'),
            ('4/6, two shifts',lambda k:error(6,4,RHO,k,2),'s--','#2563eb')]
    for label,f,style,col in curves:
        ax.plot(ks,[f(k) for k in ks],style,color=col,label=label,ms=4,lw=1.3)
    ax.set(xlabel='Number of translations k',ylabel='Worst-case total-variation error',
           title='(b) Readout bias and phase drift',ylim=(0,.65))
    ax.legend(fontsize=8,ncol=2,loc='upper left')
    fig.savefig(OUT/'figures/two-mode.pdf');fig.savefig(OUT/'figures/two-mode.png',dpi=180);plt.close(fig)
    print(json.dumps(rows,indent=2));print(json.dumps(register_rows,indent=2))
    print('PASS: independent effects, grid sums, eigennorms, witness states and j/Q nodes; residual',residual)

if __name__=='__main__':run()
