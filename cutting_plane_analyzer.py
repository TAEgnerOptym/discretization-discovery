import gurobipy as gp
from gurobipy import GRB
import time
import sys
import numpy as np

# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

# 1. Main cutting plane monitoring callback
class CutPlaneTracker(gp.Callback):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self._cuts_added = 0
        self._cuts_info = []  # Will store info about cuts
        self._last_node = -1
        self._last_time = time.time()
        
    def __call__(self):
        try:
            # Track cutting planes at the root node
            if self.where == GRB.Callback.ROOT:
                self._cuts_added = self.getIntInfo(GRB.Callback.ROOT_CUTCNT)
                print(f"Root node: {self._cuts_added} cuts added")
                
            # Track when cuts are applied (at MIP nodes)
            elif self.where == GRB.Callback.MIPNODE:
                try:
                    current_node = self.getIntInfo(GRB.Callback.MIPNODE_NODCNT)
                    
                    # Skip if we've already processed this node
                    if current_node == self._last_node:
                        return
                    
                    self._last_node = current_node
                    
                    # Only print every few nodes to avoid overwhelming output
                    if current_node % 10 == 0:
                        # Get node relaxation if available
                        if self.getIntInfo(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                            cut_count = self.getIntInfo(GRB.Callback.MIPNODE_CUTCNT)
                            if cut_count > 0:
                                print(f"Node {current_node}: {cut_count} cuts applied")
                                
                                # Try to analyze cut types if this functionality exists
                                try:
                                    # This would need access to model's internal data
                                    # Not directly available in callback
                                    pass
                                except:
                                    pass
                except Exception as e:
                    # Skip if not supported
                    pass
                
            # Track overall MIP progress
            elif self.where == GRB.Callback.MIP:
                current_time = time.time()
                if current_time - self._last_time >= 30:  # Report every 30 seconds
                    self._last_time = current_time
                    nodes = self.getIntInfo(GRB.Callback.MIP_NODCNT)
                    cuts = self.getIntInfo(GRB.Callback.MIP_CUTCNT)
                    objbst = self.getDoubleInfo(GRB.Callback.MIP_OBJBST)
                    objbnd = self.getDoubleInfo(GRB.Callback.MIP_OBJBND)
                    
                    # Calculate gap
                    if objbst == float('inf'):
                        gap = float('inf')
                    else:
                        gap = abs((objbst - objbnd) / (1e-10 + abs(objbst))) * 100
                    
                    print(f"Progress: {nodes} nodes, {cuts} cuts, Gap: {gap:.2f}%")
        except Exception as e:
            # Gracefully handle errors
            print(f"Error in callback: {str(e)}")

# Main execution
def analyze_cutting_planes():
    # Load model
    model_path = "R_104_just_flip.mps"
    print(f"Loading model from {model_path}...")
    
    with gp.Env(params=options) as env:
        with gp.read(model_path, env=env) as model:
            # Configure Gurobi to use cuts
            model.setParam("Cuts", 2)  # Aggressive cut generation
            model.setParam("CutPasses", 10)  # More cut passes
            model.setParam("PreCrush", 1)  # Allow cut generators to see the original model
            model.setParam("MIPFocus", 2)  # Focus on improving bound
            
            # Cut-specific parameters (activate specific cut types)
            cut_params = {
                "GomoryPasses": 10,  # Gomory fractional cuts
                "FlowCoverCuts": 2,  # Flow cover cuts
                "FlowPathCuts": 2,   # Flow path cuts
                "ZeroHalfCuts": 2,   # Zero-half cuts
                "MIRCuts": 2,        # Mixed integer rounding cuts
                "ModKCuts": 2,       # Mod-k cuts
                "CliqueCuts": 2,     # Clique cuts
                "CoverCuts": 2,      # Cover cuts
                "GUBCoverCuts": 2,   # GUB cover cuts
                "InfProofCuts": 2,   # Infeasibility proof cuts
                "Presolve": 2,       # Aggressive presolve
                "Symmetry": 2        # Aggressive symmetry detection
            }
            
            # Set all cut parameters
            for param, value in cut_params.items():
                try:
                    model.setParam(param, value)
                    print(f"Set {param} = {value}")
                except:
                    print(f"Parameter {param} not available")
            
            # Set other parameters
            model.setParam("OutputFlag", 1)
            model.setParam("LogToConsole", 1)
            model.setParam("DisplayInterval", 1)
            model.setParam("Threads", 1)  # Single thread for clearer reporting
            model.setParam("LogFile", "cutting_planes.log")
            model.setParam("TimeLimit", 600)  # 10-minute limit
            
            # Track variables and constraints for later analysis
            vars = model.getVars()
            constrs = model.getConstrs()
            
            print(f"Model has {len(vars)} variables and {len(constrs)} constraints")
            
            # Register our callback
            callback = CutPlaneTracker(model)
            
            print("\nStarting optimization with cut tracking...")
            model.optimize(callback)
            
            print("\nOptimization complete!")
            print(f"Status: {model.status}")
            
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                print(f"Best objective: {model.ObjVal:.6f}")
                print(f"Best bound: {model.ObjBound:.6f}")
                print(f"Gap: {model.MIPGap*100:.2f}%")
            
            print(f"Total nodes explored: {model.NodeCount}")
            print(f"Total iterations: {model.IterCount}")
            
            # ---------- ANALYZING CUTTING PLANES ----------
            print("\n===== CUTTING PLANE ANALYSIS =====")
            
            # 1. Extract cut statistics from the log
            try:
                with open("cutting_planes.log", "r") as f:
                    log_content = f.read()
                    
                # Find the cutting plane statistics section
                import re
                cut_section = re.search(r"Cutting planes:(.*?)(?:Explored \d+ nodes)", log_content, re.DOTALL)
                
                if cut_section:
                    print("Cutting planes from log:")
                    for line in cut_section.group(1).strip().split('\n'):
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"Error parsing log file: {str(e)}")
            
            # 2. Analyze the dual values of constraints
            print("\nAnalyzing constraints and their dual values:")
            try:
                # Get dual values for LP constraints (original + added cuts that are still active)
                if model.status == GRB.OPTIMAL:
                    # Get dual values (only available for continuous problems)
                    try:
                        # Try to create and solve LP relaxation to get duals
                        lp_relax = model.relax()
                        lp_relax.optimize()
                        
                        if lp_relax.status == GRB.OPTIMAL:
                            lp_constrs = lp_relax.getConstrs()
                            
                            # Find constraints with non-zero dual values
                            active_constrs = []
                            for i, constr in enumerate(lp_constrs):
                                dual = constr.Pi
                                if abs(dual) > 1e-6:  # Non-zero dual
                                    active_constrs.append((constr.ConstrName, dual))
                            
                            # Sort by absolute dual value (most significant first)
                            active_constrs.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            print(f"\nFound {len(active_constrs)} constraints with non-zero duals")
                            print("Top constraints by dual value magnitude:")
                            for name, dual in active_constrs[:20]:  # Show top 20
                                print(f"  {name}: {dual:.6f}")
                                
                            # Analyze which ones might be cuts
                            cut_prefixes = ['Gomory', 'Cover', 'Flow', 'MIR', 'GUB', 'Clique', 'Zero']
                            cuts = [c for c in active_constrs if any(p in c[0] for p in cut_prefixes)]
                            
                            if cuts:
                                print("\nIdentified cutting planes with non-zero duals:")
                                for name, dual in cuts[:20]:
                                    print(f"  {name}: {dual:.6f}")
                            else:
                                print("\nNo constraints identified as cutting planes with the standard naming patterns")
                    except Exception as e:
                        print(f"Could not analyze dual values: {str(e)}")
                else:
                    print("Model not optimal, cannot retrieve reliable dual values")
                    
                # Try to get IIS (irreducible infeasible subsystem) if the model is infeasible
                if model.status == GRB.INFEASIBLE:
                    try:
                        print("\nComputing IIS to identify critical constraints...")
                        model.computeIIS()
                        
                        if model.IISMinimal:
                            print("IIS is minimal")
                            
                        print(f"IIS contains {sum(1 for c in model.getConstrs() if c.IISConstr)} constraints")
                        
                        # List constraints in IIS
                        iis_constrs = [c for c in model.getConstrs() if c.IISConstr]
                        for c in iis_constrs[:20]:  # Show top 20
                            print(f"  {c.ConstrName}")
                    except Exception as e:
                        print(f"Could not compute IIS: {str(e)}")
            except Exception as e:
                print(f"Error analyzing dual values: {str(e)}")
            
            # 3. Write model with cuts for further analysis
            try:
                model.write("model_with_cuts.lp")
                print("\nModel with cuts written to model_with_cuts.lp")
                print("You can inspect this file to see the cutting planes that were added")
            except Exception as e:
                print(f"Could not write model with cuts: {str(e)}")
            
            # Save the solution
            model.write("solution_with_cuts.sol")
            
            print("\n===== ADDITIONAL CUTTING PLANE INFORMATION =====")
            print("1. To see more detailed cut information, check the cutting_planes.log file")
            print("2. To view specific cuts, inspect model_with_cuts.lp")
            print("3. For interactive analysis, use the Gurobi Python API to examine constraints")
            print("4. For very detailed cut analysis, consider using Gurobi's tuning tool")
            
# Run the analysis
if __name__ == "__main__":
    analyze_cutting_planes()
