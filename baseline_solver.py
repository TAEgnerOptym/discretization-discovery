import xpress as xp

from collections import defaultdict
#from src.common.route import route
from typing import Dict, DefaultDict, Set, List
import numpy as np
import pulp as pl
from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xpress as xp
import networkx as nx
import time

from scipy.sparse import csr_matrix
import pulp
from solve_gurobi_lp import solve_gurobi_milp


class baseline_solver:


    def __init__(self,full_prob,OPT_do_ILP,OPT_use_psi):
        self.BASE_times_lp_times=dict()
        self.full_prob=full_prob
        full_input_dict=full_prob.full_input_dict

        self.all_delta=full_input_dict['allDelta']
        self.all_actions=full_input_dict['allActions']
        #self.null_action:  name of the null action
        self.null_action=full_input_dict['nullAction']
        #self.all_exog; list of the names of all exogenous constraints
        self.all_exog=full_input_dict['allExogNames']
        #all_non_null_action:  list of the all_non_null_action
        self.all_non_null_action=full_input_dict['allNonNullAction']
        #self.exog_name_2_rhs:  mapping of exogenous contraitns to RHS
        self.exog_name_2_rhs=full_input_dict['exogName2Rhs']
        #self.all_integCon:  all of the contraints used to intregrate with the psi constriants 
        self.all_integCon=full_input_dict['allIntegCon']
        #self.all_primitive_vars;  list of all primitive varaiables names of psi
        self.all_primitive_vars=full_input_dict['allPrimitiveVars']
        # dictionary to mapping compact variable name to cost
        self.action_2_cost=full_input_dict['action2Cost']
        # dictionary to mapping compact variable name and constriant name to contirbutoin
        self.action_con_2_contrib=full_input_dict['actionCon2Contrib']
        #delta_con_2_contrib:   dictionary to mapping delta,constraint 2 contrib
        self.delta_con_2_contrib=full_input_dict['deltaCon2Contrib']
        #action_integCon_2_contrib
        self.action_integCon_2_contrib=full_input_dict['actionIntegCon2Contrib']
        #given any action, and interCon maps it to a contribution
        self.prim_integCon_2_contrib=full_input_dict['primIntegCon2Contrib']
        self.OPT_use_psi=OPT_use_psi
        self.OPT_do_ilp=OPT_do_ILP

        self.baseline_construct_LB_or_ILP(self.OPT_use_psi,self.OPT_do_ilp)

        if self.OPT_do_ilp==0:
            input('i not be ehere ')
            if self.full_prob.jy_opt['use_Xpress']==False:
                self.make_LP()
            else:
                self.make_xpress_LP()
            
        else:
            if self.full_prob.jy_opt['use_Xpress']==False and  self.full_prob.jy_opt['use_gurobi']==False:
                self.solve_milp()
            if self.full_prob.jy_opt['use_Xpress']==True and  self.full_prob.jy_opt['use_gurobi']==False:
                 self.solve_xpress_milp()
            if self.full_prob.jy_opt['use_gurobi']==True:
                self.call_gurobi_milp_solver()

    def baseline_construct_LB_or_ILP(self,use_psi,do_ilp):
        self.OPT_use_psi=use_psi
        self.OPT_do_ilp=do_ilp

        self.pulp_all_vars=set()
        self.dict_var_name_2_obj=dict()
        self.dict_var_name_2_is_binary=dict()
        self.dict_var_con_2_lhs_exog=dict()
        self.dict_var_con_2_lhs_eq=dict()
        self.dict_con_name_2_LB=dict()
        self.dict_con_name_2_eq=dict()
        t1=time.time()
        self.help_construct_LB_make_vars()
        self.BASE_times_lp_times['help_construct_LB_make_vars']=time.time()-t1
        t1=time.time()
        self.help_construct_UB_LB_con()
        self.BASE_times_lp_times['help_construct_UB_LB_con']=time.time()-t1
        t1=time.time()
        self.construct_constraints_exog()
        self.BASE_times_lp_times['construct_constraints_exog']=time.time()-t1
        t1=time.time()

        if use_psi==True:
            self.construct_constraints_prim()
        self.BASE_times_lp_times['construct_constraints_prim']=time.time()-t1

        #print('t_list')
        #print(t_list)
        #print(np.array(t_list)/np.sum(np.array(t_list)))
        #input('--')
    def solve_milp(self):
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_con_name_2_eq=self.dict_con_name_2_eq
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_var_name_2_is_binary=self.dict_var_name_2_is_binary
        """
        Builds and solves a MILP based on the input dictionaries.
        
        Parameters:
        dict_var_name_2_obj: dict mapping variable name (str) -> coefficient in objective.
        dict_con_name_2_LB: dict mapping inequality constraint name -> RHS lower bound for constraint (>=).
        dict_con_name_2_eq: dict mapping equality constraint name -> RHS value for the constraint.
        dict_var_con_2_lhs_exog: dict mapping (var_name, con_name) -> coefficient in the inequality constraint.
        dict_var_con_2_lhs_eq: dict mapping (var_name, con_name) -> coefficient in the equality constraint.
        dict_var_name_2_is_binary: dict mapping variable name -> 1 if binary, 0 otherwise.
        
        Returns:
        A tuple (status, objective_value, variable_values) where:
            - status: the status of the solution.
            - objective_value: the optimal value of the objective.
            - variable_values: a dict mapping variable names to their optimal values.
        """
        # Create the MILP model (for example: minimization problem)
        milp_prob = pulp.LpProblem("MILP_Problem", pulp.LpMinimize)
        
        # Create decision variables based on the input. 
        # If a variable is binary, declare it as such, otherwise as continuous (nonnegative).
        var_dict = {}
        
        for var_name, obj_coeff in dict_var_name_2_obj.items():
            if dict_var_name_2_is_binary.get(var_name, 0):
                #print('is binary')
                var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0, upBound=1, cat=pulp.LpBinary)
            else:
                var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)
        
        # Define the objective function: minimize sum(objective_coefficient * variable)
        milp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                                for var_name in dict_var_name_2_obj), "Objective"
        
        # --- Add inequality constraints (of the form: expression >= lower bound) ---
        ineq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]
        
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:
                milp_prob += expr >= dict_con_name_2_LB[con_name], con_name + "_ineq"
        
        # --- Add equality constraints ---
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]
        
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                milp_prob += expr == dict_con_name_2_eq[con_name], con_name + "_eq"
        
        # --- Solve the MILP ---
        start_time=time.time()

        solver = pulp.PULP_CBC_CMD(msg=True)
        milp_prob.solve(solver)
        end_time=time.time()
        self.milp_time=end_time-start_time
        self.milp_prob=milp_prob
        self.milp_solution = {var_name: var.varValue for var_name, var in var_dict.items()}
        self.milp_solution_status = pulp.LpStatus[milp_prob.status]
        self.milp_solution_objective_value = pulp.value(milp_prob.objective)

        #print('done ILP call')
        #input('done ILP call')
    def solve_xpress_milp(self):
        t2=time.time()
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_con_name_2_eq=self.dict_con_name_2_eq
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_var_name_2_is_binary=self.dict_var_name_2_is_binary
        import xpress as xp
        xp.init('C:/xpressmp/bin/xpauth.xpr')
        milp_prob = xp.problem("MILP_Problem")
        milp_prob.setOutputEnabled(self.full_prob.jy_opt['verbose']>0.5)

        # Create decision variables based on the input. 
        # If a variable is binary, declare it as such, otherwise as continuous (nonnegative).
        var_dict = {}

        for var_name, obj_coeff in dict_var_name_2_obj.items():
            if dict_var_name_2_is_binary.get(var_name, 0):
                var_dict[var_name] = milp_prob.addVariable(name=var_name, vartype=xp.binary)
            else:
                var_dict[var_name] = milp_prob.addVariable(name=var_name, lb=0)

        # Define the objective function: minimize sum(objective_coefficient * variable)
        objective = xp.Sum(dict_var_name_2_obj[var_name] * var_dict[var_name] 
                            for var_name in dict_var_name_2_obj)
        milp_prob.setObjective(objective, sense=xp.minimize)

        # --- Add inequality constraints (of the form: expression >= lower bound) ---
        for con_name in dict_con_name_2_LB:
            expr = xp.Sum(dict_var_con_2_lhs_exog.get((var_name, con_name), 0) * var_dict[var_name] 
                            for var_name in var_dict)
            con_ineq = xp.constraint(expr >= dict_con_name_2_LB[con_name], name=con_name )
            milp_prob.addConstraint(con_ineq)

        # --- Add equality constraints ---
        for con_name in dict_con_name_2_eq:
            expr = xp.Sum(dict_var_con_2_lhs_eq.get((var_name, con_name), 0) * var_dict[var_name] 
                            for var_name in var_dict)
            con_eq=xp.constraint(expr==dict_con_name_2_eq[con_name], name=con_name)
            milp_prob.addConstraint(con_eq)

        # --- Solve the MILP ---
        self.BASE_times_lp_times['pre_XMILP']=time.time()-t2

        start_time = time.time()
        milp_prob.solve()
        end_time = time.time()
        self.BASE_times_lp_times['XMILP']=end_time - start_time

        t3=time.time()

        self.milp_time = end_time - start_time
        self.milp_prob = milp_prob
        self.milp_solution = {var_name: milp_prob.getSolution(var_name) for var_name in var_dict}
        self.milp_solution_status = milp_prob.getProbStatus()
        self.milp_solution_objective_value = milp_prob.getObjVal()
        self.BASE_times_lp_times['post_XMILP']=time.time()-t3

    def call_gurobi_milp_solver(self):
        
        out_solution=solve_gurobi_milp(self.dict_var_name_2_obj,
                   self.dict_var_con_2_lhs_exog,
                   self.dict_con_name_2_LB,
                   self.dict_var_con_2_lhs_eq,
                   self.dict_con_name_2_eq,
                   self.dict_var_name_2_is_binary,self.full_prob.jy_opt['max_ILP_time'])
        self.milp_solution=out_solution['primal_solution']
        self.milp_solution_objective_value=out_solution['objective']
        self.BASE_times_lp_times['GUR_time_pre']=out_solution['time_pre']
        self.BASE_times_lp_times['GUR_time_opt']=out_solution['time_opt']
        self.BASE_times_lp_times['GUR_time_post']=out_solution['time_post']
        self.milp_time=out_solution['time_opt']
        self.new_actions_ignore=[]

    def make_LP(self):
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq=self.dict_con_name_2_eq
        t2=time.time()
        debug_on=False
        if debug_on==True:
            dict_con_name_2_eq=dict()
            dict_var_con_2_lhs_eq=dict()
        # --- Build the LP model ---
        my_times=[]
        t1=time.time()
        lp_prob = pulp.LpProblem("MyLP", pulp.LpMinimize)
        my_times.append(time.time()-t1)
        #print('my_times[-1];  0')
        #print(my_times[-1])
        # Create decision variables (all non-negative)
        var_dict = {}
        t1=time.time()
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)
        my_times.append(time.time()-t1)
        #print('my_times[-1];  1')
        #print(my_times[-1])
        t1=time.time()

        # Define the objective function (minimize sum(obj_coeff * var))
        lp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in self.all_actions), "Objective"
        my_times.append(time.time()-t1)
        #print('my_times[-1];  2')
        #print(my_times[-1])
        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        did_find_2=False
        #input('----')
        t1=time.time()

        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]
            if con_name=='exog_min_veh_':

                did_find_2=True
        my_times.append(time.time()-t1)
        #print('my_times[-1];  3')
        #print(my_times[-1])
        t1=time.time()
        did_find=False
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:

                if con_name=='exog_min_veh_':
                    did_find=True
                    #input('---')
                lp_prob += expr >= dict_con_name_2_LB[con_name], con_name + "_ineq"
        my_times.append(time.time()-t1)
        #print('my_times[-1];  4')
        #print(my_times[-1])
        if did_find==False:
            input('this is odd')
        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        t1=time.time()
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 5')
        #print(my_times[-1])
        t1=time.time()
        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                lp_prob += expr == dict_con_name_2_eq[con_name], con_name #+ "_eq"
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 6')
        #print(my_times[-1])     
                #input('----')
        # --- Solve the LP ---
        # Using the default CBC solver here.
        self.BASE_times_lp_times['pre_lp_solve']=time.time()-t2
        start_time=time.time()
        solver = pulp.PULP_CBC_CMD(msg=False)
        lp_prob.solve(solver)
        end_time=time.time()
        self.lp_time=end_time-start_time
        my_times.append(end_time-start_time)
        self.BASE_times_lp_times['lp_time']=end_time-start_time
        t3=time.time()
        #print('my_times[-1]; 7')
        #print(my_times[-1])     
        self.lp_prob=lp_prob
        self.lp_primal_solution=dict()
        t1=time.time()

        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name]=var.varValue
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 8')
        #print(my_times[-1])     
        self.lp_status=pulp.LpStatus[lp_prob.status]
        self.lp_objective= pulp.value(lp_prob.objective)
        t1=time.time()
        self.lp_dual_solution=dict()
        for con_name, constraint in lp_prob.constraints.items():
            self.lp_dual_solution[con_name]=constraint.pi
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 9')
        #print(my_times[-1])     
        self.BASE_times_lp_times['post_lp_time']=time.time()-t3
        #print(np.array(my_times)/(np.sum(np.array(my_times))))
        #input('---')
        print('self.BASE_times_lp_times')
        print(self.BASE_times_lp_times)
        print('--')
        total = sum(self.BASE_times_lp_times.values())
        time_percentage_LP = {key: (val / total if total != 0 else 0) for key, val in self.BASE_times_lp_times.items()}
        print('time_percentage_LP')
        print(time_percentage_LP)
        print('----')
        if self.lp_status=='Infeasible':
            input('HOLD')
    def make_xpress_LP(self):
        #/Users/julian/Documents/FICO\ Xpress\ Config/xpauth.xpr
        #xp.init('/Users/julian/Documents/FICO\ Xpress\ Config/xpauth.xpr')
        xp.init(self.full_prob.jy_opt['xpress_file_loc'])
        t2=time.time()
        dict_var_name_2_obj = self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog = self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB = self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq = self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq = self.dict_con_name_2_eq

        debug_on = False
        if debug_on == True:
            dict_con_name_2_eq = dict()
            dict_var_con_2_lhs_eq = dict()

        # --- Build the LP model ---
        lp_prob = xp.problem("MyLP")
        lp_prob.setOutputEnabled(self.full_prob.jy_opt['verbose']>0.5)

        # Create decision variables (all non-negative)
        var_dict = {}
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = xp.var(name=var_name, lb=0)

        # Define the objective function (minimize sum(obj_coeff * var))
        objective = xp.Sum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj)
        lp_prob.setObjective(objective, sense=xp.minimize)

        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        did_find_2 = False
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            if con_name not in ineq_expressions:
                ineq_expressions[con_name] = 0
            ineq_expressions[con_name] += coeff * var_dict[var_name]
            if con_name == 'exog_min_veh_':
                did_find_2 = True

        did_find = False
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:
                if con_name == 'exog_min_veh_':
                    did_find = True
                lp_prob.addConstraint(expr >= dict_con_name_2_LB[con_name], name=con_name + "_ineq")

        if did_find == False:
            input('this is odd')

        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            if con_name not in eq_expressions:
                eq_expressions[con_name] = 0
            eq_expressions[con_name] += coeff * var_dict[var_name]

        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                lp_prob.addConstraint(expr == dict_con_name_2_eq[con_name], name=con_name)

        # --- Solve the LP ---
        self.BASE_times_lp_times['pre_XP_lp']=time.time()-t2
        start_time = time.time()
        lp_prob.solve()
        end_time = time.time()

        self.lp_time = end_time - start_time
        self.BASE_times_lp_times['XP_lp_time']=self.lp_time
        t3=time.time()
        self.lp_prob = lp_prob
        self.lp_primal_solution = dict()
        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name] = var.getSolution()

        self.lp_status = lp_prob.getProbStatus()
        self.lp_objective = lp_prob.getObjVal()
        self.lp_dual_solution = dict()

        for con_name in lp_prob.getConstraints():
            self.lp_dual_solution[con_name] = lp_prob.getDual(con_name)

        if self.lp_status == 'Infeasible':
            input('HOLD')
        self.BASE_times_lp_times['post_XLP']=time.time()-t3


    def help_construct_LB_make_vars(self):
        use_psi=self.OPT_use_psi
        do_ilp=self.OPT_do_ilp
        self.dict_var_name_2_is_binary=defaultdict(int)
        self.names_binary=[]
        if use_psi==True and do_ilp==True:
            for var_name in self.all_primitive_vars:
                self.dict_var_name_2_obj[var_name]=0
                self.dict_var_name_2_is_binary[var_name]=1
        
        my_x_typr='Binary'
        if do_ilp==False or use_psi==True:
            my_x_typr='Continuous'
        for var_name in self.all_non_null_action:
            self.dict_var_name_2_obj[var_name]=self.action_2_cost[var_name]
        if do_ilp==True or use_psi==False:
            for var_name in self.all_non_null_action:
                self.dict_var_name_2_is_binary[var_name]=1
        for var_name in self.all_delta:
            self.dict_var_name_2_obj[var_name]=0
        if self.full_prob.jy_opt['all_vars_binary']==True:
            for var_name in set(self.dict_var_name_2_obj)-set(self.all_delta):
                self.dict_var_name_2_is_binary[var_name]=1
    def help_construct_UB_LB_con(self):
        
        t1=time.time()
        for exog_name in self.exog_name_2_rhs:
            self.dict_con_name_2_LB[exog_name]=self.exog_name_2_rhs[exog_name]
            
        if self.OPT_use_psi==True and self.OPT_do_ilp==True:
            for con_name in self.all_integCon:
                self.dict_con_name_2_eq[con_name]=0
        
    def construct_constraints_exog(self):
        for v_con in self.delta_con_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_exog[tuple([var_name,con_name])]=self.delta_con_2_contrib[v_con]
        
        for v_con in self.action_con_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_exog[tuple([var_name,con_name])]=self.action_con_2_contrib[v_con]
    
    def construct_constraints_prim(self):
        for v_con in self.prim_integCon_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_eq[tuple([var_name,con_name])]=self.prim_integCon_2_contrib[v_con]
        for v_con in self.action_integCon_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_eq[tuple([var_name,con_name])]=self.action_integCon_2_contrib[v_con]
    