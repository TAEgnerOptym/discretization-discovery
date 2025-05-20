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

# Define callback function to monitor branching decisions
def branchCallback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # Called when MIP solution is found
        try:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            nodecnt = model.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)
            print(f"**** New solution found at node {nodecnt} with objective {obj:.6f} ****")
        except AttributeError:
            # Alternative format for older Gurobi versions
            print("**** New MIP solution found ****")
        
    elif where == gp.GRB.Callback.MIP:
        # More compatible way to track branch-and-bound progress
        try:
            nodecnt = model.cbGet(gp.GRB.Callback.MIP_NODCNT)
            objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            solcnt = model.cbGet(gp.GRB.Callback.MIP_SOLCNT)
            
            # Only print every 10 nodes to avoid cluttering the output
            if nodecnt % 10 == 0:
                gap = float('inf') if objbst == float('inf') else abs((objbst - objbnd) / (1e-10 + abs(objbst))) * 100
                print(f"Node: {nodecnt}, Best obj: {objbst:.6f}, Bound: {objbnd:.6f}, Gap: {gap:.2f}%, Solutions: {solcnt}")
        except Exception as e:
            print(f"Error tracking MIP progress: {str(e)}")
            pass

    # We can't get the specific branching decisions in all Gurobi versions,
    # but we can track which nodes are being processed in the branch-and-bound tree

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
        
        # Log file for detailed analysis
        model.setParam("LogFile", "gurobi_branching_detailed.log")
        
        # Save node file information (compatible with all Gurobi versions)
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