import xpress as xp

# Create an empty problem
lp = xp.problem()

# Create variables with upper bounds
x1 = xp.var(name='x1', lb=0, ub=5)
x2 = xp.var(name='x2', lb=0, ub=3)
x3 = xp.var(name='x3', lb=0, ub=4)

# Add variables to model
lp.addVariable(x1, x2, x3)

# Add a simple constraint
lp.addConstraint(x1 + 2 * x2 + x3 <= 8)

# Set an objective
lp.setObjective(x1 + 4*x2 + 2*x3, sense=xp.maximize)

# Solve the LP
lp.solve()
#
reduced_costs = lp.getAttrib('rc', [x1, x2, x3])
# upper_bound_duals = lp.getAttrib(xp.getubdual, [x1, x2, x3])
print("\nUpper Bound Duals:")
for var, ubdual in zip([x1, x2, x3], upper_bound_duals):
    print(f"{var.name}: Upper bound dual = {ubdual}")
