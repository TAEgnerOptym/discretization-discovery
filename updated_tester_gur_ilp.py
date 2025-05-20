import gurobipy as gp
from gurobipy import GRB
import time

# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

# Define callback function to monitor branching decisions
def branchCallback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # Called when MIP solution is found
        obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
        nodecnt = model.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)
        print(f"**** New solution found at node {nodecnt} with objective {obj:.6f} ****")
        
    elif where == gp.GRB.Callback.MIPNODE:
        # Called when a node in the branch-and-bound tree is processed
        status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            node_count = model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)
            obj_val = model.cbGet(gp.GRB.Callback.MIPNODE_OBJVAL)
            print(f"Node {node_count}: Relaxation objective {obj_val:.6f}")
            
            # Get selected branching variable (only works when Gurobi has chosen a variable)
            try:
                branchvar = model.cbGet(gp.GRB.Callback.MIPNODE_BRVAR)
                if branchvar >= 0:  # Valid variable index
                    vars = model.getVars()
                    print(f"Branching on variable {vars[branchvar].VarName}")
            except:
                pass  # Gurobi might not have chosen a variable yet

model_path="R_104_just_flip.mps"

with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        # Set parameters for detailed branching information
        model.setParam("OutputFlag", 1)          # Enable solver output
        model.setParam("LogToConsole", 1)        # Display log in console
        model.setParam("DisplayInterval", 1)     # Update display frequently
        
        # Detailed branching parameters
        model.setParam("VarBranch", 1)           # Strong branching (recommended for viewing branching decisions)
        model.setParam("BranchDir", 0)           # Down branch first at each node
        model.setParam("Threads", 1)             # Single thread for clearer logs
        
        # Parameters to control level of detail
        model.setParam("MIPFocus", 3)            # Focus on bound improvement
        model.setParam("Tree", 1)                # Store the branch-and-bound tree
        
        # Log file for detailed analysis
        model.setParam("LogFile", "gurobi_branching_detailed.log")
        
        # Save branch-and-bound tree information
        model.setParam("NodefileDir", ".")       # Directory for node files
        model.setParam("NodefileStart", 0.5)     # Start writing node files at 0.5 GB
        
        # List binary variables
        bin_set=[]
        for var in model.getVars():
             if var.VType == GRB.BINARY:
                bin_set.append(var)
                
        print('Number of binary variables:', len(bin_set))
        print('Variable names of the first 10 binary variables:')
        for i in range(min(10, len(bin_set))):
            print(f"  {bin_set[i].VarName}")

        # Keep your existing parameter settings
        model.setParam("Cuts", 0)                # Disable all cutting planes
        model.setParam("CutPasses", 0)           # No passes even beyond root
        
        model.update()
        
        # Add a time limit if needed to avoid excessive computation
        # model.setParam("TimeLimit", 600)  # 10-minute limit
        
        # Print header for the optimization process
        print("\n==== Starting optimization with branching information ====\n")
        
        # Optimize with callback for detailed branching information
        model.optimize(branchCallback)
        
        # Print summary statistics after optimization
        print("\n==== Optimization completed ====")
        print(f"Status: {model.status}")
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            print(f"Best objective: {model.ObjVal:.6f}")
            print(f"Bound: {model.ObjBound:.6f}")
            print(f"Gap: {model.MIPGap*100:.2f}%")
        print(f"Total nodes explored: {model.NodeCount}")
        print(f"Total iterations: {model.IterCount}")
        
        # Save the model with solution information
        model.write("solution_with_branching.sol")