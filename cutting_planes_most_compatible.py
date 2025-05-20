import gurobipy as gp
from gurobipy import GRB
import time
import os

# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

def analyze_cuts_compatible():
    """
    Very simplified approach to analyze cutting planes
    that works with all Gurobi versions.
    """
    # Load model
    model_path = "R_104_just_flip.mps"
    print(f"Loading model from {model_path}...")
    
    with gp.Env(params=options) as env:
        # Step 1: Solve LP relaxation to get baseline
        print("\nSolving LP relaxation first...")
        try:
            model_lp = gp.read(model_path, env=env)
            
            # Relax integrality
            for var in model_lp.getVars():
                if var.VType != GRB.CONTINUOUS:
                    var.VType = GRB.CONTINUOUS
            
            model_lp.update()
            model_lp.setParam("LogFile", "lp_relaxation.log")
            model_lp.optimize()
            
            lp_obj = model_lp.ObjVal if model_lp.status == GRB.OPTIMAL else None
            print(f"LP relaxation objective: {lp_obj}")
            
            # Save LP model
            model_lp.write("lp_model.lp")
            model_lp.write("lp_model.sol")
            
            # Release memory
            model_lp.dispose()
            
        except Exception as e:
            print(f"Error solving LP relaxation: {str(e)}")
            lp_obj = None
        
        # Step 2: Solve with cuts at root only
        print("\nSolving with cuts at root node only...")
        try:
            model_root = gp.read(model_path, env=env)
            
            # Set parameters for root-only cuts
            model_root.setParam("Cuts", 2)           # Aggressive cut generation
            model_root.setParam("CutPasses", 50)     # Many cut passes
            model_root.setParam("NodeLimit", 1)      # Stop after root node
            model_root.setParam("LogFile", "root_cuts.log")
            
            # Solve
            model_root.optimize()
            
            # Get root relaxation with cuts
            root_obj = model_root.ObjBound
            print(f"Root relaxation with cuts objective: {root_obj}")
            
            # Calculate improvement from cuts
            if lp_obj is not None:
                improvement = ((root_obj - lp_obj) / abs(lp_obj)) * 100
                print(f"Improvement from cuts: {improvement:.2f}%")
            
            # Save model with cuts
            model_root.write("root_cuts.lp")
            
            # Release memory
            model_root.dispose()
            
        except Exception as e:
            print(f"Error solving with root cuts: {str(e)}")
        
        # Step 3: Solve full MIP with cuts
        print("\nSolving full MIP with cuts...")
        try:
            model = gp.read(model_path, env=env)
            
            # Parameters for cut generation
            model.setParam("Cuts", 2)                # Aggressive cut generation
            model.setParam("CutPasses", 10)          # Cut passes
            model.setParam("LogFile", "mip_cuts.log")
            model.setParam("TimeLimit", 300)         # 5-minute limit
            
            # Activate specific cut types
            try:
                model.setParam("GomoryPasses", 10)   # More Gomory cuts
                model.setParam("MIRCuts", 2)         # Aggressive MIR cuts
                model.setParam("ZeroHalfCuts", 2)    # Aggressive zero-half cuts
                model.setParam("FlowCoverCuts", 2)   # Aggressive flow cover cuts
                model.setParam("CoverCuts", 2)       # Aggressive cover cuts
            except:
                print("Some cut parameters not supported in this Gurobi version")
            
            # Optimize
            model.optimize()
            
            # Print results
            print("\nMIP solution:")
            print(f"Status: {model.status}")
            
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                print(f"Best objective: {model.ObjVal}")
                print(f"Best bound: {model.ObjBound}")
                print(f"Gap: {model.MIPGap*100:.2f}%")
            
            print(f"Nodes explored: {model.NodeCount}")
            
            # Save model with all cuts
            model.write("mip_cuts.lp")
            model.write("mip_cuts.sol")
            
        except Exception as e:
            print(f"Error solving MIP with cuts: {str(e)}")
        
        # Step 4: Analyze logs to extract cut information
        print("\n===== ANALYZING CUTTING PLANE INFORMATION =====")
        
        # Process logs to find cutting plane info
        log_files = ["root_cuts.log", "mip_cuts.log"]
        cut_counts = {}
        
        for log_file in log_files:
            if not os.path.exists(log_file):
                continue
                
            print(f"\nAnalyzing {log_file}...")
            
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                
                # Look for the cutting plane section
                import re
                cut_section = re.search(r"Cutting planes:(.*?)(?:Explored \d+ nodes|\Z)", log_content, re.DOTALL)
                
                if cut_section:
                    print("Cutting planes applied:")
                    for line in cut_section.group(1).strip().split('\n'):
                        line = line.strip()
                        print(f"  {line}")
                        
                        # Parse counts
                        parts = line.split(':')
                        if len(parts) == 2:
                            cut_type = parts[0].strip()
                            count = int(parts[1].strip())
                            cut_counts[cut_type] = cut_counts.get(cut_type, 0) + count
                else:
                    print("No cutting plane section found")
            except Exception as e:
                print(f"Error analyzing log file: {str(e)}")
        
        # Show total cuts by type
        if cut_counts:
            print("\nTotal cuts by type:")
            for cut_type, count in sorted(cut_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cut_type}: {count}")
        
        # Step 5: Compare models to identify added constraints
        print("\nComparing models to identify cutting planes...")
        
        try:
            # Count constraints in each model
            files_to_compare = [("lp_model.lp", "LP relaxation"), 
                               ("root_cuts.lp", "Root with cuts"),
                               ("mip_cuts.lp", "MIP with cuts")]
            
            constraint_counts = {}
            
            for file_name, description in files_to_compare:
                if not os.path.exists(file_name):
                    continue
                    
                # Count constraints by finding lines that start with "subject to"
                try:
                    with open(file_name, "r") as f:
                        content = f.read()
                    
                    # Find the constraints section
                    constr_section = re.search(r"subject to(.*?)(?:Bounds|Binary variables|\Z)", content, re.DOTALL)
                    
                    if constr_section:
                        # Count constraints (lines that contain an equality or inequality)
                        constraints = [line for line in constr_section.group(1).split('\n') 
                                      if '>=' in line or '<=' in line or '=' in line]
                        count = len(constraints)
                        constraint_counts[description] = count
                        print(f"  {description}: {count} constraints")
                except Exception as e:
                    print(f"  Error analyzing {file_name}: {str(e)}")
            
            # Compare constraint counts
            if len(constraint_counts) >= 2:
                items = list(constraint_counts.items())
                for i in range(1, len(items)):
                    desc1, count1 = items[i-1]
                    desc2, count2 = items[i]
                    diff = count2 - count1
                    print(f"  Added constraints from {desc1} to {desc2}: {diff}")
            
        except Exception as e:
            print(f"Error comparing models: {str(e)}")
        
        print("\n===== ADDITIONAL INFORMATION =====")
        print("To identify active cutting planes by dual values:")
        print("1. Load the LP relaxation or root node model")
        print("2. Solve to optimality")
        print("3. Examine the dual values (Pi) of each constraint")
        print("4. Constraints with non-zero duals are active in the solution")
        print("5. Compare constraint names to identify which are cutting planes")
        
if __name__ == "__main__":
    analyze_cuts_compatible()
