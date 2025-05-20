import gurobipy as gp
from gurobipy import GRB
import time
from itertools import product
# Set desired solver options
import math

def count_non_101_coefficients_tol(model, tol=1e-6):
    count = 0
    for constr in model.getConstrs():
        row = model.getRow(constr)
        for i in range(row.size()):
            coeff = row.getCoeff(i)
            if not any(math.isclose(coeff, x, abs_tol=tol) for x in (-1.0, 0.0, 1.0)):
                count += 1
    print(f"Number of coefficients not approximately -1, 0, or 1: {count}")

    if count>0:
        input('ok this is very wrong looking ')

    return count

def count_non_101_rhs_constraints(model):
    count = 0
    for constr in model.getConstrs():
        rhs = constr.RHS  # Correct way to access the RHS
        if rhs not in {-1, 0, 1}:
            count += 1
    print(f"Number of constraints with RHS not in {{-1, 0, 1}}: {count}")
    if count>0:
        input('wrong rhs counter')
    return count
def configure_cut_types(model, lift, mir, flowcover, zerohalf):
    model.setParam("LiftProjectCuts", lift)
    model.setParam("MIRCuts", mir)
    model.setParam("FlowCoverCuts", flowcover)
    model.setParam("ZeroHalfCuts", zerohalf)


def run_root_relaxation_for_cut_combinations(model):
    cut_types = ['Lift', 'MIR', 'FlowCover', 'ZeroHalf']
    combinations = list(product([0, 2], repeat=4))  # 2^4 = 16 combinations

    results = []

    for combo in combinations:
        lift, mir, flowcover, zerohalf = combo

        # Clone model to preserve original
        cloned_model = model.copy()

        # Set cut parameters
        #configure_cut_types(cloned_model, lift, mir, flowcover, zerohalf)

        # Disable all branching — solve only LP relaxation
        #cloned_model.setParam("Presolve", 2)         # Keep presolve on
        #cloned_model.setParam("Cuts", 0)             # Turn off automatic cut control
        #cloned_model.setParam("Heuristics", 0)       # No primal heuristics
        #cloned_model.setParam("PrePasses", 10)       # Allow preprocessing to generate cuts
        cloned_model.setParam("MIPFocus", 3)         # Focus on bound improvement
        cloned_model.setParam("NodeLimit", 1)        # Prevent branching
        #cloned_model.setParam("OutputFlag", 1)       # Suppress solver output
        configure_cut_types(cloned_model, lift, mir, flowcover, zerohalf)
        #cloned_model.setParam("LiftProjectCuts", 0)
        #cloned_model.setParam("MIRCuts", 0)
        #cloned_model.setParam("FlowCoverCuts", 0)
        #cloned_model.setParam("ZeroHalfCuts", 2)
        cloned_model.update()
        #print('starting')
        cloned_model.optimize()
        #input('---')
       ## print('cloned_model.Status')
       # print(cloned_model.Status)
        #print('cloned_model.ObjBound')
        #print(cloned_model.ObjBound)
        #input('done')
        bound = cloned_model.ObjBound #if cloned_model.Status == gp.GRB.OPTIMAL else None

        results.append({
            'Cuts': combo,
            'Bound': bound
        })

    return results

def clean_model(model):
    model.update()
    
        
    delta_vars = [v for v in model.getVars() if "delta" in v.VarName]
    count = len(delta_vars)
    print(f"Number of variables with 'delta' in their name: {count}")
    

    constraints_to_remove = set()

    for constr in model.getConstrs():
        row = model.getRow(constr)
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            if "delta" in var.VarName and abs(coeff) > 1e-10:
                constraints_to_remove.add(constr)
                break  # No need to check the rest of the row

    for constr in constraints_to_remove:
        model.remove(constr)

        # Phase 2: Remove all delta variables
    for v in delta_vars:
        model.remove(v)
    model.update()

    constraints_to_remove = []

    for constr in model.getConstrs():
        rhs = constr.RHS
        if rhs not in {-1, 0, 1}:
            constraints_to_remove.append(constr)
    
    for constr in constraints_to_remove:
        model.remove(constr)
    
    model.update()
    print(f"Number of constraints with RHS not in {{-1, 0, 1}}: {count}")

    count = 0
    for constr in model.getConstrs():
        row = model.getRow(constr)
        for i in range(row.size()):
            coeff = row.getCoeff(i)
            if coeff not in {-1.0, 0.0, 1.0}:
                count += 1

    print(f"Number of coefficients not equal to -1, 0, or 1: {count}")

def force_integer(model):
    for var in model.getVars():
        if  var.VType != gp.GRB.BINARY and "delta" not in var.VarName:
            var.VType = gp.GRB.INTEGER

options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }
#model_path="../Optym_gurobi_files_R107_LP/phase_1_file_num8417073.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_458.mps"
model_path="../optym_gurobi_file_ILP/C_104_25_10.mps"
model_path="../optym_gurobi_file_ILP/C_104_50_25.mps"
model_path="../optym_gurobi_file_ILP/R_104_50.mps"
model_path="../optym_gurobi_file_ILP/R_104_50.mps"
model_path="../optym_gurobi_file_ILP/C_104_100_10.mps"
model_path="FANCY_COMPRESS_model_name.mps"
model_path="NO_FANCY_COMPRESS_model_name.mps"
model_path="R_104_MPS_NO_FANCY.mps"
model_path="R_104_NO_FancY_2.mps"
#model_path="R_104_MPS_NO_FANCY_2.mps"
model_path="R104_super_fine.mps"
#model_path="tryMe.mps"
#model_path="R_104_fany.mps"
#model_path="R_104_just_flip.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_25_compress.mps"
do_clean=1
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.setParam("DisplayInterval", 1)
        #model.setParam("MIPFocus", 3)
        if do_clean>0.5:
            clean_model(model)
            model.update()
            count_non_101_rhs_constraints(model)
            count_non_101_coefficients_tol(model)
        #model.optimize()
        if 1>0:
            #clean_model(model)

            force_integer(model)
            #model.setParam("Cuts", 0)
            model.setParam("NodeLimit", 100000)        # Prevent branching
            #model.setParam("Gomery",0)
            #model.setParam("MIRCuts", 0)
            #model.setParam("FlowCoverCuts", 0)
            model.setParam("ZeroHalfCuts", 2)
            model.update()
            input('IM HERE RIGHT')
            model.optimize()
            input('hihih')
        results=run_root_relaxation_for_cut_combinations(model)
        for entry in results:
            lift, mir, flow, zero = entry['Cuts']
            print(f"Lift={lift}, MIR={mir}, FlowCover={flow}, ZeroHalf={zero} → Root bound = {entry['Bound']}")
        print('do clean')
        print(do_clean)