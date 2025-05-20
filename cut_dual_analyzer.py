import gurobipy as gp
from gurobipy import GRB
import os
import re
import csv

# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

def extract_active_cuts():
    """
    Extract cut constraints with significant dual values
    """
    print("Analyzing cutting planes and their dual values...")
    
    # 1. First, create a list of original constraints from the initial model
    try:
        model_path = "R_104_just_flip.mps"
        original_model = None
        original_constrs = set()
        
        with gp.Env(params=options) as env:
            original_model = gp.read(model_path, env=env)
            for c in original_model.getConstrs():
                original_constrs.add(c.ConstrName)
            
            print(f"Original model has {len(original_constrs)} constraints")
            original_model.dispose()
    except Exception as e:
        print(f"Error reading original model: {str(e)}")
        original_constrs = set()
    
    # 2. Check if mip_cuts.lp exists (from previous run)
    if not os.path.exists("mip_cuts.lp"):
        print("mip_cuts.lp not found. Run the cutting plane analysis script first.")
        return
    
    # 3. Load model with cuts and relax for dual analysis
    print("\nLoading model with cuts and creating LP relaxation...")
    
    with gp.Env(params=options) as env:
        try:
            model = gp.read("mip_cuts.lp", env=env)
            
            # Count constraints
            all_constrs = model.getConstrs()
            print(f"Model with cuts has {len(all_constrs)} constraints")
            
            # Identify potential cutting planes (constraints not in original model)
            potential_cuts = []
            for c in all_constrs:
                if c.ConstrName not in original_constrs:
                    potential_cuts.append(c.ConstrName)
            
            print(f"Found {len(potential_cuts)} potential cutting planes")
            
            # Create LP relaxation
            for var in model.getVars():
                if var.VType != GRB.CONTINUOUS:
                    var.VType = GRB.CONTINUOUS
            
            model.update()
            
            # Solve LP relaxation to get dual values
            print("\nSolving LP relaxation to get dual values...")
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                print("LP relaxation solved optimally")
                
                # Extract dual values
                active_cuts = []
                for c in all_constrs:
                    dual = c.Pi
                    
                    # Consider constraints with significant dual values
                    if abs(dual) > 1e-6:
                        is_cut = c.ConstrName not in original_constrs
                        active_cuts.append({
                            'constraint': c.ConstrName,
                            'dual': dual,
                            'is_cut': is_cut
                        })
                
                # Sort by absolute dual value
                active_cuts.sort(key=lambda x: abs(x['dual']), reverse=True)
                
                # Display active cuts
                print("\nConstraints with active dual values:")
                count = 0
                for cut in active_cuts:
                    cut_label = "[CUT]" if cut['is_cut'] else "[ORIGINAL]"
                    print(f"{cut_label} {cut['constraint']}: {cut['dual']:.6f}")
                    count += 1
                    if count >= 20:  # Show top 20
                        print("...")
                        break
                
                # Save active cuts to CSV
                with open("active_cuts.csv", "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['constraint', 'dual', 'is_cut'])
                    for cut in active_cuts:
                        writer.writerow([cut['constraint'], cut['dual'], cut['is_cut']])
                
                print(f"\nSaved {len(active_cuts)} active constraints to active_cuts.csv")
                
                # Additional analysis - group by cut type
                cut_types = {}
                for cut in active_cuts:
                    if not cut['is_cut']:
                        continue
                        
                    name = cut['constraint']
                    
                    # Try to identify cut type from name
                    cut_type = "Unknown"
                    type_patterns = {
                        'Gomory': ['gomory', 'gmi'],
                        'Cover': ['cover'],
                        'Flow': ['flow'],
                        'MIR': ['mir'],
                        'Clique': ['clique'],
                        'Zero-half': ['zero', 'zh'],
                        'GUB': ['gub']
                    }
                    
                    for type_name, patterns in type_patterns.items():
                        if any(p in name.lower() for p in patterns):
                            cut_type = type_name
                            break
                    
                    if cut_type not in cut_types:
                        cut_types[cut_type] = []
                    
                    cut_types[cut_type].append((name, cut['dual']))
                
                print("\nActive cutting planes by type:")
                for cut_type, cuts in cut_types.items():
                    print(f"{cut_type}: {len(cuts)} cuts")
                    
                    # Sum of absolute dual values for this type
                    total_impact = sum(abs(dual) for _, dual in cuts)
                    print(f"  Total impact: {total_impact:.6f}")
                    
                    # Show top cuts of each type
                    cuts.sort(key=lambda x: abs(x[1]), reverse=True)
                    for i, (name, dual) in enumerate(cuts[:3]):  # Top 3
                        print(f"  {name}: {dual:.6f}")
                    
                    if len(cuts) > 3:
                        print(f"  ... ({len(cuts)-3} more)")
                
                # Try to identify which cutting planes are associated with binary variables
                # that have value 1 in the MIP solution
                try:
                    # Load MIP solution if available
                    active_binary_vars = set()
                    
                    if os.path.exists("mip_cuts.sol"):
                        with open("mip_cuts.sol", "r") as f:
                            for line in f:
                                if "# " not in line and line.strip():  # Not a comment
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        var_name = parts[0]
                                        value = float(parts[1])
                                        if abs(value - 1.0) < 1e-6:  # Value is 1
                                            active_binary_vars.add(var_name)
                        
                        print(f"\nFound {len(active_binary_vars)} active binary variables in MIP solution")
                        
                        # Try to find cuts that involve these variables
                        print("\nAnalyzing relationship between active cuts and active variables...")
                        
                        # Extract constraint expressions from LP file
                        constraint_exprs = {}
                        if os.path.exists("mip_cuts.lp"):
                            with open("mip_cuts.lp", "r") as f:
                                content = f.read()
                            
                            # Find constraints section
                            constr_match = re.search(r"subject to(.*?)(?:Bounds|Binary variables|\Z)", content, re.DOTALL)
                            if constr_match:
                                constr_section = constr_match.group(1)
                                
                                # Process each constraint
                                current_name = None
                                current_expr = ""
                                
                                for line in constr_section.split('\n'):
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    # Check if this line starts a new constraint
                                    name_match = re.match(r'^(\S+):\s+(.*)$', line)
                                    if name_match:
                                        # Save previous constraint if any
                                        if current_name is not None:
                                            constraint_exprs[current_name] = current_expr
                                        
                                        # Start new constraint
                                        current_name = name_match.group(1)
                                        current_expr = name_match.group(2)
                                    else:
                                        # Continue previous constraint
                                        if current_name is not None:
                                            current_expr += " " + line
                                
                                # Save last constraint
                                if current_name is not None:
                                    constraint_exprs[current_name] = current_expr
                        
                        # Find active cuts that involve active binary variables
                        active_var_cuts = []
                        for cut in active_cuts:
                            if not cut['is_cut']:
                                continue
                                
                            name = cut['constraint']
                            if name in constraint_exprs:
                                expr = constraint_exprs[name]
                                
                                # Check if any active binary var appears in this cut
                                for var_name in active_binary_vars:
                                    # Look for variable name with word boundaries
                                    if re.search(r'\b' + re.escape(var_name) + r'\b', expr):
                                        active_var_cuts.append((name, cut['dual'], var_name))
                                        break
                        
                        if active_var_cuts:
                            print("\nActive cuts involving variables with value 1 in solution:")
                            for name, dual, var_name in active_var_cuts[:10]:  # Show top 10
                                print(f"  {name} (dual: {dual:.6f}) involves {var_name}")
                            
                            if len(active_var_cuts) > 10:
                                print(f"  ... ({len(active_var_cuts)-10} more)")
                        else:
                            print("No direct relationship found between active cuts and active binary variables")
                    else:
                        print("MIP solution file not found. Cannot analyze relationship with active variables.")
                    
                except Exception as e:
                    print(f"Error analyzing relationship with active variables: {str(e)}")
                
            else:
                print(f"LP relaxation could not be solved optimally. Status: {model.status}")
            
        except Exception as e:
            print(f"Error analyzing cuts: {str(e)}")

if __name__ == "__main__":
    extract_active_cuts()
