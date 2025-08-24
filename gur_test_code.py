import gurobipy as gp
import numpy as np
model_path="model_name.mps"
branch_file="branch_priorities.txt"

#model_path="CONTROL_RC_102_model_name.mps"
#branch_file="CONTROL_RC_102_branch_priorities.txt"


#model_path="Treatment_RC_102_model_name.mps"
#branch_file="Treatment_RC_102__branch_priorities.txt"

#branch_file="TREAT_C_104_50_branch_priorities.txt"
#model_path="TREAT_C_104_50_model_name.mps"


#branch_file="CONTROL_C_104_50_branch_priorities.txt"
#model_path="CONTROL_C_104_50_model_name.mps"

#branch_file="TREAT_C104_100_branch_priorities.txt"
#model_path="TREAT_C104_100_model_name.mps"
options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }

#observe this parameter min_priority_keep_integer.  
#
# I made the following table to show how following my proposed branching order performs.  We will only keep ineger varaibles as integer if their priority is above this value
# note  that Binary varaibles are set to have upper bound 1 if they are relaxed to be non-binary.  The aim here is to show the value of this branching order.  However I think that if it is used convergence should be a lot faster.  

# 
# #min_priority_keep_integer;  OBJECTIVE; Number of integer variables
# 1001: 8.184388879e+02 ; 0 integer 
# 1000:  821.387 ; 1 integer (all binary)
# 999: 821.857  ; 2 integer (all binary)
# 998: 822.029   ; 3 integer (all binary)
# 997:  822.379   ; 4 integer (all binary)
# 996:  822.412  ; 5 integer (all binary)
# 995:  822.4999200  ; 6 integer  (all binary)
# 994:  822.7  ; 7 integer  (all binary)
#993:  822.9 ;8 integer (all binary)
#40:  822.9 ;1340 integer (all binary)
#-infty; 822.9;   43824 integer (41664 binary)
min_priority_keep_integer=0


with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        with open(branch_file, "r") as f:
            for line in f:
                name, bp = line.strip().split()
                var = model.getVarByName(name)
                if int(bp) < min_priority_keep_integer:#<min_priority_keep_integer or int(bp)>max_priority_keep_integer:
                    #if var.vType==gp.GRB.BINARY and var.Ub>=1:
                    #    var.Ub=1
                    var.vType=gp.GRB.CONTINUOUS
                else:
                    var.BranchPriority=int(bp)
                #if  int (bp)>990 and int(bp)<997:
                #    var.vType=gp.GRB.CONTINUOUS
            #tmp1=model.getVarByName('v58501')
            #tmp2=model.getVarByName('v58501')
            #print('[tmp1.LB,tmp1.UB]')
            #print([tmp1.LB,tmp1.UB])
            #print('[tmp2.LB,tmp2.UB]')
            #print([tmp2.LB,tmp2.UB])
            #print('---')
                #if int(bp)==1000:
                #    var.UB=0
                #if int(bp)==999:
                #    var.UB=0
                #if int(bp)==998:
                #    var.UB=0
            #model.setParam("Cuts", 0)                # Disable all cutting planes
            ##model.setParam("Presolve", 0)                # Disable all cutting planes
            #model.setParam("CutPasses", 0)           # No passes even beyond root
            #model.setParam("MIPFocus", 3)
            side_ineq_use_constraints = {
                constr.ConstrName: constr
                for constr in model.getConstrs()
                if constr.ConstrName.startswith("side_ineq_use")
            }
            for con in model.getConstrs():
                print(con.ConstrName)
            print('side_ineq_use_constraints')
            print(side_ineq_use_constraints)
            print("Complete constraints (side_ineq_use*)")
            for name, constr in side_ineq_use_constraints.items():
                lhs = model.getRow(constr)   # linear expression
                sense = constr.Sense
                rhs = constr.RHS
                print(f"{name}: {lhs} {sense} {rhs}")
            print('side_ineq_use_constraints')
            input('---')
            print(side_ineq_use_constraints)
            model.update()
            model.optimize()

            if 1<0:
                solution_values = {var.VarName: var.X for var in model.getVars()}

                model.reset()
                for var in model.getVars():
                    if var.BranchPriority>90:
                        #var.Start=solution_values[var.VarName]
                        var.LB=solution_values[var.VarName]
                        var.UB=solution_values[var.VarName]
                print('resetting ')
                model.update()

                model.optimize()