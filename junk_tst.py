import xpress as xp

# Create a simple problem
model = xp.problem()

# Add two variables
x1 = xp.var(name='x1', lb=0, ub=10)
x2 = xp.var(name='x2', lb=0, ub=5)
model.addVariable(x1, x2)

# Add a constraint
model.addConstraint(x1 + x2 <= 10)

# Set objective
model.setObjective(x1 + 2 * x2, sense=xp.maximize)

# Solve the model
model.solve()
print(model.getObjVal)
#print("Original solution:")
#print(f"x1 = {x1.value}, x2 = {x2.value}")

# Now change the upper bound of x1
model.chgbounds(['x1'],['U'], [1])

# Re-solve the model
model.solve()

print("\nSolution after changing x1's upper bound to 3:")
print(f"x1 = {x1.value}, x2 = {x2.value}")
