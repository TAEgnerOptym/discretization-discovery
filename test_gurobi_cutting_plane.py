import gurobipy as gp
from gurobipy import GRB
import time
# Set desired solver options

import gurobipy as gp

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse.linalg import inv

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def remove_delta_and_cons(model):

    delta_constraints = []
    delta_vars = [v for v in model.getVars() if "delta" in v.VarName]

    # Step 1: Identify delta constraints and store their data
    for constr in model.getConstrs():
        row = model.getRow(constr)
        for i in range(row.size()):
            if "delta" in row.getVar(i).VarName:
                delta_constraints.append(constr)
                break

    delta_con_data = []
    for constr in delta_constraints:
        expr = model.getRow(constr)
        rhs = constr.RHS
        sense = constr.Sense
        name = constr.ConstrName
        delta_con_data.append((expr, sense, rhs, name))
        model.remove(constr)
    for var in delta_vars:
        model.remove(var)
    print('len(delta_constraints)')
    print(len(delta_constraints))
    model.update()
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def generate_gomory_cuts_all_integer(model, tol=1e-5, max_cuts=10, verbose=True):
    model.update()
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print("Model not solved to optimality.")
        return []

    vars_list = model.getVars()
    x_vals = model.getAttr("X")

    A=model.getA()
    vbasis = model.getAttr("VBasis")
    basis_var_indices = [i for i, status in enumerate(vbasis) if status == GRB.BASIC]
    #assert len(basis_var_indices) == A.shape[0], "Number of basic vars ≠ number of constraints!"
    print('len(basis_var_indices)')
    print(len(basis_var_indices))
    B = A[:, basis_var_indices] 
    print('B.shape')
    print(B.shape)
    print('A.shape')
    print(A.shape)
    print('model.getVars()')
    print(len(model.getVars()))
    print('model.getConstrs()')
    print(len(model.getConstrs()))
    input('')
    cuts = []
    cut_count = 0

    for var, x_val, basis_status in zip(vars_list, x_vals, vbasis):
        if basis_status != GRB.BASIC:
            continue
        if abs(x_val - round(x_val)) <= tol:
            continue

        f0 = x_val - np.floor(x_val)
        row = model.getRow(var)
        coeff_dict = {}

        for j in range(row.size()):
            v = row.getVar(j)
            coeff = row.getCoeff(j)
            fj = coeff - np.floor(coeff)
            if abs(fj) > tol:
                coeff_dict[v] = fj

        if coeff_dict:
            cuts.append((coeff_dict, f0))
            cut_count += 1
            if verbose:
                print(f"Generated Gomory cut for {var.VarName} (value: {x_val:.4f})")

        if cut_count >= max_cuts:
            break

    return cuts

def gomory_cutting_plane_all_integer(model, tol=1e-5, max_cuts_per_iter=10, max_iters=20, verbose=True):
    model.update()

    # Relax all variables to continuous
    for v in model.getVars():
        v.VType = GRB.CONTINUOUS
    model.update()

    for iteration in range(max_iters):
        cuts = generate_gomory_cuts_all_integer(model, tol=tol, max_cuts=max_cuts_per_iter, verbose=verbose)

        if not cuts:
            if verbose:
                print(f"Iteration {iteration + 1}: No violated Gomory cuts. Terminating.")
            break

        for i, (coeff_dict, rhs) in enumerate(cuts):
            expr = gp.LinExpr()
            for v, coef in coeff_dict.items():
                expr += coef * v
            model.addConstr(expr >= rhs, name=f"gomory_cut_{iteration}_{i}")

        model.update()

        if verbose:
            print(f"Iteration {iteration + 1}: Added {len(cuts)} cuts.")

    return model

options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }


model_path="R104_super_fine.mps"

#model_path="model_name.mps"
#model_path="R_112_model_name.mps"
do_remove=False
do_bring_back=False
do_force_integer=True
do_make_linear=False
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.setParam("DisplayInterval", 1)
        print('changing options')
        model.setParam("Method", 1)
        #model.setParam("BranchDir", 1)
        model.setParam("Presolve", 0)
        remove_delta_and_cons(model)


        constrs = model.getConstrs()
        rhs_vals = model.getAttr("RHS", constrs)
        senses = model.getAttr("Sense", constrs)

        # Identify constraints to keep and those to delete
        keep_constrs = []
        delete_constrs = []

        for constr, rhs in zip(constrs, rhs_vals):
            if rhs in (-1, 0, 1):
                keep_constrs.append(constr)
            else:
                delete_constrs.append(constr)
                row = model.getRow(constr)
                lhs_str = " + ".join(
                    f"{row.getCoeff(i):.4g}*{row.getVar(i).VarName}"
                    for i in range(row.size())
                )
                rhs = constr.RHS
                name=constr.ConstrName
                print(f"{name}: {lhs_str} {constr.Sense} {rhs:.4g}\n")
                input('--')
        # Step 2: Remove unwanted constraints
        for constr in delete_constrs:
            model.remove(constr)

        # Step 3: Change all remaining constraints to equalities
        model.update()
        for constr in keep_constrs:
            constr.setAttr("Sense", "=")

        model.update()

        constrs = model.getConstrs()
        senses = model.getAttr("Sense", constrs)

        eq_count = sum(1 for s in senses if s == '=')
        leq_count = sum(1 for s in senses if s == '<')
        geq_count = sum(1 for s in senses if s == '>')

        print(f"Equality constraints (=):       {eq_count}")
        print(f"Inequality constraints (≤):     {leq_count}")
        print(f"Inequality constraints (≥):     {geq_count}")
        print(f"Total constraints:              {len(constrs)}")
        #input('hihi')

        constrs = model.getConstrs()
        senses = model.getAttr("Sense", constrs)
        names = model.getAttr("ConstrName", constrs)

        

        for constr, sense, name in zip(constrs, senses, names):
            if sense in ('<', '>'):
                row = model.getRow(constr)
                lhs_str = " + ".join(
                    f"{row.getCoeff(i):.4g}*{row.getVar(i).VarName}"
                    for i in range(row.size())
                )
                rhs = constr.RHS
                print(f"{name}: {lhs_str} {sense} {rhs:.4g}\n")
        input('---')
        gomory_cutting_plane_all_integer(model)
        #model.optimize()
        #input('---')
        #model.update()
        #solve_with_lazy_delta_constraints(model,do_remove,do_bring_back,do_force_integer,do_make_linear)
         

        #model.optimize()