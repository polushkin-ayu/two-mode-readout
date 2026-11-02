"""Operational comparison: K rotations followed by one B-outcome digit readout.

All normalized inputs on |m|<=M are tested via the exact worst-case TV norm.
Grid models receive ideal encoding/permutations (optimistic for those models).
Spectral rotations use rounded single-qubit phase angles; same 256-point
midpoint readout, exactly representable for phase precision >=9 bits.
"""
import json
from decimal import Decimal, getcontext
from math import gcd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

getcontext().prec=70
RHO=(Decimal(5).sqrt()-1)/2
OUT=Path(__file__).resolve().parents[1]/'results'/'extension'
OUT.mkdir(parents=True,exist_ok=True)

M=4; B=8; D=256
MODES=np.arange(-M,M+1)
DIFF=MODES[None,:]-MODES[:,None]
MASKS=np.array([[(s>>b)&1 for b in range(B)] for s in range(2**(B-1))])


def exact_effects():
    out=[]
    for b in range(B):
        # Integral exp(2 pi i (n-m) theta) over [b/B,(b+1)/B].
        out.append(np.exp(2j*np.pi*DIFF*(b+.5)/B)*np.sinc(DIFF/B)/B)
    return np.array(out)


EXACT=exact_effects()
EXACT_SUB=np.einsum('sb,bij->sij',MASKS,EXACT)


def grid_effects(Q,offset=.5):
    theta=(np.arange(Q)+offset)/Q
    f=np.exp(2j*np.pi*np.outer(theta,MODES))/np.sqrt(Q)
    bins=np.floor(B*theta).astype(int)
    return np.array([f[bins==b].conj().T@f[bins==b] for b in range(B)])


def tv_error(subsets, phase_errors, k, witness=False):
    z=np.exp(2j*np.pi*k*phase_errors)
    matrices=subsets*z.conj()[None,:,None]*z[None,None,:]-EXACT_SUB
    eig=np.linalg.eigvalsh(matrices)
    err=float(np.max(abs(eig)))
    if not witness:return err
    si,ei=np.unravel_index(np.argmax(abs(eig)),eig.shape)
    _,vec=np.linalg.eigh(matrices[si])
    return err,vec[:,ei],int(si)


def scalar_delta(a,Q):
    return MODES*float(Decimal(a)/Q-RHO)


def rounded_spectral(bits,k=1):
    # Signed 8-bit encoding of m. Eight local phase gates per rotation.
    weights=[2**r for r in range(7)]+[-128]
    angles=[(RHO*k*w*2**bits).to_integral_value(rounding='ROUND_HALF_EVEN')/Decimal(2**bits) for w in weights]
    errors=[]
    for m in MODES:
        j=int(m)%D
        phase=sum((angles[r] for r in range(8) if j&(1<<r)),Decimal(0))
        delta=phase-int(m)*RHO*k
        errors.append(float(delta))
    return np.array(errors)


def evaluate(label,Q,phase_errors,K=64,shifts=1):
    effects=np.mean([grid_effects(Q,(r+.5)/shifts) for r in range(shifts)],axis=0)
    if shifts>1:
        assert np.max(abs(effects-grid_effects(Q*shifts)))<1e-12
    subsets=np.einsum('sb,bij->sij',MASKS,effects)
    errors=[tv_error(subsets,phase_errors,k) for k in range(K+1)]
    k=int(np.argmax(errors))
    e,c,subset=tv_error(subsets,phase_errors,k,True)
    z=np.exp(2j*np.pi*k*phase_errors)
    p=np.real(np.einsum('i,bij,j->b',c.conj(),EXACT,c))
    actual=np.real(np.einsum('i,bij,j->b',(z*c).conj(),effects,z*c))
    assert abs(np.sum(abs(p-actual))/2-e)<2e-12
    row={'label':label,'Q':Q,'K':K,'shifts':shifts,'max_TV':e,'worst_k':k,'readout_only_TV':errors[0],
         'worst_state_real':c.real.tolist(),'worst_state_imag':c.imag.tolist(),
         'target_probabilities':p.tolist(),'approx_probabilities':actual.tolist(),
         'curve':errors}
    return row


def run():
    assert np.max(abs(EXACT.sum(axis=0)-np.eye(2*M+1)))<1e-12
    assert np.linalg.eigvalsh(EXACT).min()>-1e-12
    result={'M':M,'B':B,'D_capacity':D,'K':64,'grids':[],'spectral':[],'shift_averaged':[],'compiled_spectral':[]}
    for a,Q in [(158,256),(144,233),(89,144)]:
        r=evaluate(f'{a}/{Q}',Q,scalar_delta(a,Q))
        r.update({'a':a,'slope_error':float(abs(Decimal(a)/Q-RHO))})
        result['grids'].append(r)
        print('GRID',r['label'],r['max_TV'],r['readout_only_TV'],flush=True)
    for bits in [10,12,14,16,18,20]:
        r=evaluate(f'{bits} phase bits',D,rounded_spectral(bits))
        r['phase_bits']=bits
        result['spectral'].append(r)
        print('SPECTRAL',bits,r['max_TV'],flush=True)
    result['spectral_exact']=evaluate('Exact spectral phases',D,np.zeros(2*M+1))
    for a,Q in [(158,256),(144,233),(89,144)]:
        r=evaluate(f'{a}/{Q}, 8 shifted readouts',Q,scalar_delta(a,Q),shifts=8)
        result['shift_averaged'].append(r)
        print('SHIFT_AVERAGED',r['label'],r['max_TV'],r['readout_only_TV'],flush=True)
    for bits in [10,12,14,16]:
        sub=np.einsum('sb,bij->sij',MASKS,grid_effects(D))
        errors=[tv_error(sub,rounded_spectral(bits,k),1) for k in range(65)]
        row={'phase_bits':bits,'max_TV':max(errors),'worst_k':int(np.argmax(errors)),
             'phase_gates_per_query':8,'curve':errors}
        result['compiled_spectral'].append(row)
        print('COMPILED_SPECTRAL',bits,row['max_TV'],flush=True)
    # Exact POVM on the 9-dimensional input: no dimension-dependent obstruction
    # to a single terminal readout. Dilation size at most B*(2M+1)=72.
    roots=[]
    for e in EXACT:
        values,vectors=np.linalg.eigh(e)
        roots.append((vectors*np.sqrt(np.maximum(values,0)))@vectors.conj().T)
    isometry=np.concatenate(roots,axis=0)
    result['exact_POVM_dilation']={'rows':len(isometry),'columns':2*M+1,
         'isometry_error':float(np.max(abs(isometry.conj().T@isometry-np.eye(2*M+1))))}
    assert result['exact_POVM_dilation']['isometry_error']<1e-12
    # Scan all denominators at endpoints K=0,8,64. Endpoint objective only;
    # no claim that this minimizes maximum-over-time error.
    scans=[]
    for Q in range(2*M+1,D+1):
        a=int((RHO*Q).to_integral_value(rounding='ROUND_HALF_EVEN'))
        sub=np.einsum('sb,bij->sij',MASKS,grid_effects(Q))
        delta=scalar_delta(a,Q)
        scans.append({'Q':Q,'a':a,'TV_k0':tv_error(sub,delta,0),
                      'TV_k8':tv_error(sub,delta,8),'TV_k64':tv_error(sub,delta,64)})
    result['denominator_scan']=scans
    result['best_endpoint']={key:min(scans,key=lambda x:x[key]) for key in ['TV_k0','TV_k8','TV_k64']}
    # Independent continuous quadrature at a witness: verify exact effect formula.
    row=result['grids'][1]
    c=np.array(row['worst_state_real'])+1j*np.array(row['worst_state_imag'])
    from scipy.integrate import quad
    integrated=[]
    for b in range(B):
        integrated.append(quad(lambda x:abs(np.sum(c*np.exp(2j*np.pi*MODES*x)))**2,b/B,(b+1)/B,
                               epsabs=1e-12,epsrel=1e-12)[0])
    result['quadrature_error']=float(np.max(abs(np.array(integrated)-row['target_probabilities'])))
    assert result['quadrature_error']<1e-12
    # Uniform Fourier mode provides a readout-only lower bound for Q=233.
    effects=grid_effects(233)
    uniform=effects[:,M,M].real
    result['uniform_readout_TV_233']=float(np.sum(abs(uniform-1/B))/2)
    assert abs(result['uniform_readout_TV_233']-7/(8*233))<1e-12
    (OUT/'readout-results.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result)
    print('BEST_ENDPOINT',result['best_endpoint'],flush=True)
    print('EXACT_SPECTRAL',result['spectral_exact']['max_TV'],flush=True)
    print('DILATION',result['exact_POVM_dilation'],flush=True)


def plot(result):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for row in result['grids']:
        axs[0].semilogy(range(65),row['curve'],label='Grid '+row['label'])
    row=result['shift_averaged'][1]
    axs[0].semilogy(range(65),row['curve'],label='144/233, 8 readout shifts')
    for bits in [12,16]:
        row=next(r for r in result['spectral'] if r['phase_bits']==bits)
        axs[0].semilogy(range(65),row['curve'],'--',label=f'Spectral, {bits}-bit phases')
    axs[0].axhline(.01,color='black',ls=':',label='1% error')
    axs[0].set(xlabel='Number of rotations k',ylabel='Worst-case TV error of probabilities',title='One three-digit readout, all inputs |m|≤4')
    axs[0].legend(fontsize=8)
    rows=result['spectral']
    axs[1].semilogy([r['phase_bits'] for r in rows],[r['max_TV'] for r in rows],'o-',label='Spectral scheme')
    axs[1].axhline(result['grids'][1]['max_TV'],ls='--',label='Ideal grid 144/233')
    axs[1].axhline(result['shift_averaged'][1]['max_TV'],ls='-.',label='144/233, 8 shifts')
    axs[1].axhline(result['spectral_exact']['max_TV'],ls=':',label='Limit of the chosen readout')
    axs[1].set(xlabel='Bits of precision per phase angle',ylabel='Worst-case TV error for k≤64',title='Control-precision check with 8 data qubits')
    axs[1].legend(fontsize=8)
    fig.savefig(OUT/'readout.png',dpi=180)
    fig.savefig(OUT/'readout.pdf')
    plt.close(fig)


if __name__=='__main__':run()
