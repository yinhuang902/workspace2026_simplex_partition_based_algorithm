"""
run_st_rv2_case.py -- Simplex runner for SNGO-master/Global/st_rv2

20 variables, 10 <= constraints, concave quadratic objective.
Objective: Min Sum(-ai*xi^2 - bi*xi) with varying coefficients per variable.
All constraints are linear <=. 9 constraints with nonzero RHS + sum<=200.

Aligned with problem_st_rv2.py (snoglode):
  nfirst=5, nparam=5, seed=1234.  Smoke mode uses nscen=10.
  c1-c5 RHS perturbed via PlasmoOld addnoise: ub + |ub|*U(0,2) for nonzero,
  U(0,10) for zero.  Scenario 0 is unperturbed.

Intentional difference vs problem_st_rv2.py:
  First-stage variables (x1-x5) carry explicit FBBT-tightened upper bounds
  required by the simplex search box.  These do not reduce the feasible region.
"""
import argparse
from pathlib import Path
from time import perf_counter
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937, addnoise_le

_RHS_BASE = [("c1_rhs",405.),("c2_rhs",450.),("c3_rhs",430.),("c4_rhs",360.),
             ("c5_rhs",315.),("c6_rhs",330.),("c7_rhs",350.),("c8_rhs",245.),
             ("c9_rhs",390.),("c10_rhs",200.)]

def create_model():
    m = pyo.ConcreteModel()
    # FBBT-tightened bounds: x_i <= min over constraints j of (RHS_j / coeff_ij)
    # x1<=390/5=78 (c9), x2<=330/8=41.25 (c6), x3<=350/8=43.75 (c7),
    # x4<=350/7=50 (c7), x5<=245/7=35 (c8)
    _fs_ub = {1: 78.0, 2: 41.25, 3: 43.75, 4: 50.0, 5: 35.0}
    for i in range(1, 6): setattr(m, f"x{i}", pyo.Var(bounds=(0, _fs_ub[i]), initialize=0))
    for i in range(6, 21): setattr(m, f"x{i}", pyo.Var(bounds=(0, None), initialize=0))
    for i,(name,val) in enumerate(_RHS_BASE):
        setattr(m, name, pyo.Param(mutable=True, initialize=val))
    x=[getattr(m,f"x{i}") for i in range(1,21)]

    m.c1 =pyo.Constraint(expr=6*x[0]+2*x[1]+4*x[2]+3*x[4]+4*x[5]+9*x[6]+5*x[8]+x[9]+9*x[10]+6*x[11]+7*x[13]+9*x[14]+2*x[15]+8*x[17]+2*x[18]+4*x[19]<=m.c1_rhs)
    m.c2 =pyo.Constraint(expr=6*x[0]+5*x[1]+x[2]+8*x[3]+4*x[5]+3*x[6]+9*x[7]+6*x[9]+4*x[10]+7*x[11]+5*x[12]+2*x[14]+5*x[15]+8*x[16]+9*x[18]+8*x[19]<=m.c2_rhs)
    m.c3 =pyo.Constraint(expr=8*x[1]+6*x[2]+2*x[3]+6*x[4]+4*x[6]+4*x[7]+6*x[8]+9*x[10]+4*x[11]+6*x[12]+9*x[13]+9*x[15]+9*x[16]+3*x[17]+x[19]<=m.c3_rhs)
    m.c4 =pyo.Constraint(expr=8*x[0]+7*x[2]+3*x[3]+2*x[4]+x[5]+7*x[7]+4*x[8]+7*x[9]+3*x[11]+4*x[12]+x[13]+6*x[14]+2*x[16]+8*x[17]+9*x[18]<=m.c4_rhs)
    m.c5 =pyo.Constraint(expr=x[0]+5*x[1]+5*x[3]+5*x[4]+x[5]+3*x[6]+5*x[8]+7*x[9]+4*x[10]+6*x[12]+x[13]+3*x[14]+4*x[15]+3*x[17]+5*x[18]+5*x[19]<=m.c5_rhs)
    m.c6 =pyo.Constraint(expr=x[0]+8*x[1]+7*x[2]+x[4]+6*x[5]+x[6]+6*x[7]+7*x[9]+3*x[10]+6*x[11]+4*x[13]+6*x[14]+x[15]+4*x[16]+x[18]+4*x[19]<=m.c6_rhs)
    m.c7 =pyo.Constraint(expr=5*x[1]+8*x[2]+7*x[3]+3*x[5]+3*x[6]+8*x[7]+6*x[8]+6*x[10]+4*x[11]+3*x[12]+4*x[14]+2*x[15]+5*x[16]+2*x[17]+4*x[19]<=m.c7_rhs)
    m.c8 =pyo.Constraint(expr=x[0]+3*x[2]+2*x[3]+7*x[4]+2*x[6]+x[7]+x[8]+7*x[9]+4*x[11]+3*x[12]+5*x[13]+3*x[15]+6*x[16]+3*x[17]+x[18]<=m.c8_rhs)
    m.c9 =pyo.Constraint(expr=5*x[0]+5*x[1]+2*x[3]+x[4]+9*x[5]+7*x[7]+4*x[8]+8*x[9]+5*x[10]+2*x[12]+4*x[13]+4*x[14]+4*x[16]+8*x[17]+9*x[18]+x[19]<=m.c9_rhs)
    m.c10=pyo.Constraint(expr=sum(x)<=m.c10_rhs)

    # Objective: concave quadratic with varying coefficients (exact from Julia)
    # -ai*xi^2 - bi*xi
    a = [0.00015,0.00245,0.00095,0.0038,0.0029,0.0024,0.0034,0.0018,0.00305,0.00025,
         0.00195,0.0008,0.0035,0.0027,0.002,0.0026,0.0048,0.00275,0.00235,0.00275]
    b = [0.0051,0.2205,0.0171,0.6384,0.435,0.4704,0.4556,0.2916,0.0549,0.0245,
         0.3588,0.1456,0.672,0.5184,0.016,0.1404,0.2592,0.418,0.1081,0.264]
    m.obj_expr = sum(-a[i]*x[i]**2 - b[i]*x[i] for i in range(20))
    return m

def build_models(nscen=10,nfirst=5,nparam=5,seed=1234,**kw):
    rng=JuliaMT19937(seed); ml=[]; fl=[]; mx=nparam
    for s in range(nscen):
        m=create_model(); av=[getattr(m,f"x{i}") for i in range(1,21)]; f=av[:nfirst]
        if s>0:
            for idx in range(min(mx,len(_RHS_BASE))):
                p,bv=_RHS_BASE[idx]; getattr(m,p).set_value(addnoise_le(bv,rng))
        ml.append(m); fl.append(f)
    return ml,fl

# smoke: 10-scenario aligned path matching problem_st_rv2.py (snoglode)
MODE_PARAMS={"smoke":{"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,"plot_output_dir":"results/st_rv2_smoke/plots","output_csv_path":"results/st_rv2_smoke/simplex_result.csv"},
 "full":{"nscen":100,"target_nodes":300,"gap_stop_tol":1e-5,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":60.,"plot_every":None,"plot_output_dir":"results/st_rv2_full/plots","output_csv_path":"results/st_rv2_full/simplex_result.csv"}}
BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-3, "TimeLimit": 30}
Q_MAX = 1e10

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("smoke","full"),default="full"); ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--nfirst",type=int,default=5)
    args=ap.parse_args(); nf=args.nfirst; cfg=dict(MODE_PARAMS[args.mode])
    out=Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True,exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True,exist_ok=True)
    print("="*60+f"\nst_rv2 -- nfirst={nf}, nparam=5, seed={args.seed}\n"+"="*60)
    t0=perf_counter(); ml,fl=build_models(cfg["nscen"],nfirst=nf,nparam=5,seed=args.seed); S=len(ml)
    bb=[BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb=[MSBundle(ml[s],fl[s],options=BUNDLE_OPTIONS) for s in range(S)]
    res=run_pid_simplex_3d(model_list=ml,first_vars_list=fl,base_bundles=bb,ms_bundles=mb,target_nodes=cfg["target_nodes"],min_dist=1e-3,gap_stop_tol=cfg["gap_stop_tol"],time_limit=cfg["time_limit"],enable_ef_ub=cfg["enable_ef_ub"],ef_time_ub=cfg["ef_time_ub"],plot_every=cfg["plot_every"],plot_output_dir=cfg["plot_output_dir"],output_csv_path=str(out),enable_3d_plot=False,axis_labels=tuple(f"x{i+1}" for i in range(nf)))
    t1=perf_counter(); LB=res.get("LB_hist",[]); UB=res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")

if __name__=="__main__": main()
