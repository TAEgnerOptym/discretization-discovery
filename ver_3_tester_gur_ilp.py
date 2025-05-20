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
        try:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            nodecnt = model.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)
            print(f"**** New solution found at node {nodecnt} with objective {obj:.6f} ****")
        except AttributeError:
            # Alternative format for older Gurobi versions
            print("**** New MIP solution found ****")
        
    elif where == gp.GRB.Callback.MIPNODE:
        # Called when a node in the branch-and-bound tree is processed
        try:
            # Try to get node count - different versions use different constants
            node_count = -1
            try:
                node_count = model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)
            except AttributeError:
                try:
                    node_count = model.cbGet(gp.GRB.Callback.MIP_NODCNT)
                except:
                    pass
            
            # Only print info periodically to avoid overwhelming logs
            if node_count % 10 == 0 and node_count >= 0:
                print(f"Processing node {node_count}")
                
                # Try to get node solution if possible
                try:
                    nodeSol = model.cbGetNodeRel()
                    # Find some fractional binary variables that might be branching candidates
                    vars = model.getVars()
                    print(f"Potential branching candidates at node {node_count}:")
                    count = 0
                    for i, var in enumerate(vars):
                        if var.VType == GRB.BINARY and i < len(nodeSol):
                            val = nodeSol[i]
                            # Only show variables with fractional values
                            if 0.01 < val < 0.99:
                                print(f"  {var.VarName}: {val:.6f}")
                                count += 1
                                if count >= 5:  # Show at most 5 fractional variables
                                    break
                    if count == 0:
                        print("  No fractional binary variables found")
                except:
                    print("  Could not access node solution")
        except Exception as e:
            print(f"Error in callback: {str(e)}")
            pass  # Skip if not available in this Gurobi version

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