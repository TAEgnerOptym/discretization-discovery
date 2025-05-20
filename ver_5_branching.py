import gurobipy as gp
from gurobipy import GRB
import time
import sys

# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

# Define simple callback to track branching
def branchCallback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # Called when MIP solution is found
        try:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            nodecnt = model.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)
            print(f"**** New solution found at node {nodecnt} with objective {obj:.6f} ****")
        except AttributeError:
            print("**** New MIP solution found ****")

    elif where == gp.GRB.Callback.MIP:
        # Track nodes processed
        try:
            nodecount = model.cbGet(gp.GRB.Callback.MIP_NODCNT)
            if nodecount % 100 == 0 and nodecount > 0:  # Every 100 nodes
                objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                print(f"Node {nodecount}: Best={objbst:.2f}, Bound={objbnd:.2f}, " +
                      f"Gap={100.0*abs(objbst-objbnd)/(1e-10+abs(objbst)):.2f}%")
        except:
            pass

model_path = "R_104_just_flip.mps"

# Small test model - enable to test with a small example
create_test_model = False
if create_test_model:
    try:
        # Create a small test model
        test_model = gp.Model("test")
        x = test_model.addVars(5, 5, vtype=GRB.BINARY, name="x")
        test_model.setObjective(gp.quicksum(i*j*x[i,j] for i in range(5) for j in range(5)), GRB.MINIMIZE)
        test_model.addConstrs((gp.quicksum(x[i,j] for j in range(5)) == 1 for i in range(5)), name="c1")
        test_model.write("small_test.mps")
        model_path = "small_test.mps"
        print("Created test model")
    except Exception as e:
        print(f"Could not create test model: {str(e)}")

# Try adding parameters directly to the environment
try:
    print("\nAttempting to set parameters in environment")
    env = gp.Env()
    env.setParam("OutputFlag", 1)
    env.setParam("LogToConsole", 1)
    env.setParam("LogFile", "gurobi_branching_direct.log")
    env.setParam("VarBranch", -1)  # Basic branching (NOT strong branching)
    env.setParam("Cuts", 0)
    env.setParam("CutPasses", 0)
    env.setParam("DisplayInterval", 1)
    env.setParam("Threads", 1)
    
    print("Loading model from: " + model_path)
    try:
        direct_model = gp.read(model_path, env)
        print("Model loaded with parameters set in environment")
        direct_model.optimize()
        print("Direct optimization complete")
    except Exception as e:
        print(f"Error in direct optimization: {str(e)}")
except Exception as e:
    print(f"Could not set environment parameters: {str(e)}")

# Now try with regular approach
print("\nAttempting with standard parameter setting")
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        print("Model loaded, setting parameters individually")
        # Basic parameters
        model.setParam("OutputFlag", 1)
        model.setParam("LogToConsole", 1)
        model.setParam("DisplayInterval", 1)
        model.setParam("Threads", 1)
        
        # Enable basic branching (NOT strong branching)
        model.setParam("VarBranch", 0)  # Use basic branching strategy (more visible in logs)
        model.setParam("BranchDir", 0)  # Down branch first at each node
        
        # Log file setup
        model.setParam("LogFile", "gurobi_branching_detailed.log")
        
        # Node file parameters
        model.setParam("NodefileDir", ".")
        model.setParam("NodefileStart", 0.5)
        
        # Focus on bound improvement
        model.setParam("MIPFocus", 3)
        model.setParam("Presolve", 0)  # Disable presolve to see more variables
        
        # Disable cuts
        model.setParam("Cuts", 0)
        model.setParam("CutPasses", 0)
        
        # Try to specifically analyze branching decisions
        try:
            model.setParam("PreSolve", 0)      # Disable presolve completely
            model.setParam("Symmetry", 0)      # Disable symmetry detection 
            model.setParam("Aggregate", 0)     # Disable aggregation in presolve
            model.setParam("Heuristics", 0)    # Disable heuristics to force branching
            model.setParam("ImproveStartTime", 0)  # Start improving immediately
            # Add verbose MIP parameters
            model.setParam("MIPFocus", 1)      # Focus on finding feasible solutions
            model.setParam("VarBranch", 0)     # Use basic branching (not strong)
            print("Extra branching analysis parameters set")
        except Exception as e:
            print(f"Could not set all branching analysis parameters: {str(e)}")
        
        model.update()
        
        # Print model information 
        print("\n===== MODEL INFORMATION =====")
        print(f"Number of variables: {model.NumVars}")
        print(f"Number of binary variables: {sum(1 for v in model.getVars() if v.VType == GRB.BINARY)}")
        print(f"Number of constraints: {model.NumConstrs}")
        print("============================\n")
        
        # Print version information
        print("\n===== GUROBI VERSION INFO =====")
        print(f"Gurobi Version: {gp.gurobi.version()}")
        print(f"Python Version: {sys.version}")
        print("===============================\n")
        
        # Count binary variables
        bin_set = []
        for var in model.getVars():
            if var.VType == GRB.BINARY:
                bin_set.append(var)
        print(f'Number of binary variables: {len(bin_set)}')
        
        # Show first few binary variables
        if len(bin_set) > 0:
            print('Variable names of some binary variables:')
            for i in range(min(10, len(bin_set))):
                print(f"  {bin_set[i].VarName}")
        
        # Print optimization settings
        print("\n==== Starting optimization with branching information ====")
        print("Parameter settings:")
        for param_name in ["VarBranch", "BranchDir", "MIPFocus", "Cuts", "Presolve", "Threads"]:
            try:
                value = model.getParamInfo(param_name)
                print(f"  {param_name}: {value}")
            except:
                print(f"  {param_name}: <not available>")
        print("\n")
        
        # Use a time limit to avoid excessive computation
        try:
            model.setParam("TimeLimit", 600)  # 10-minute limit
            print("Time limit set to 10 minutes")
        except:
            print("Could not set time limit")
        
        # Choose a more direct way to see branching in the log
        # 1. First try to dump detailed output
        try:
            model.setParam("MIPTrace", 2)  # Most detailed MIP trace
            print("MIPTrace parameter set to most detailed level")
        except:
            print("MIPTrace parameter not available")
        
        # Try to send branching trace to a log file
        try:
            # Set dump file for debugging
            model.write("model_before_solve.lp")
            print("Model written to model_before_solve.lp")
        except Exception as e:
            print(f"Could not write model: {str(e)}")
        
        # Optimize with callback
        try:
            print("\nStarting optimization...")
            model.optimize(branchCallback)
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
        
        # Print summary statistics
        print("\n==== Optimization completed ====")
        print(f"Status: {model.status}")
        
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            print(f"Best objective: {model.ObjVal:.6f}")
            print(f"Bound: {model.ObjBound:.6f}")
            print(f"Gap: {model.MIPGap*100:.2f}%")
            
        print(f"Total nodes explored: {model.NodeCount}")
        print(f"Total iterations: {model.IterCount}")
        
        # Try to save solution information
        try:
            model.write("final_solution.sol")
            print("Solution saved to final_solution.sol")
        except Exception as e:
            print(f"Could not write solution: {str(e)}")

print("\nScript completed.")