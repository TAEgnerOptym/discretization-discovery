import gurobipy as gp
from gurobipy import GRB
import time
import random
# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }
#model_path="../Optym_gurobi_files_R107_LP/phase_1_file_num8417073.mps"
model_path="../Optym_gurobi_files_C104_LP/phase_1_file_num7953247.mps"
model_path="../Optym_gurobi_files_C104_LP/phase_1_file_num4668991.mps"
#model_path="../Optym_gurobi_files_C104_LP/phase_1_file_num8073826.mps"
#model_path="../Optym_gurobi_files_C104_LP/phase_1_file_num1068033.mps"
#model_path="../Optym_gurobi_files_C104_LP/phase_1_file_num9945700.mps"
#model_path="../Optym_gurobi_files_R104_LP/phase_1_file_num5071590.mps"
#model_path="../Optym_gurobi_files_R104_LP/phase_1_file_num7905750.mps"
#model_path="../Optym_gurobi_files_C103_LP/phase_1_file_num5807368.mps"
#model_path="../Optym_gurobi_files_C103_LP/phase_1_file_num9492258.mps"
#model_path="../Optym_gurobi_files_C103_100_LP/phase_1_file_num8412538.mps"
#model_path="../Optym_gurobi_files_C104_LP/phase_1_file_num4668991.mps"
#model_path="../Optym_gurobi_files_C103_100_LP/phase_1_file_num8412538.mps"
model_path="../Optym_gurobi_files_C104_LP/phase_1_file_num1068033.mps"
the_filename = model_path.rsplit("phase_1_file_num", 1)[-1]

with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:

        print('Running Phase One')
        print('Running Phase One')
        print('Running Phase One')
        print('Running Phase One')
        print('Running Phase One')
        print('Running Phase One')
        model.write("INPUT_"+the_filename+"_phase1.mps") 

        time_LP_phase_1=time.time()

        model.optimize()
        time_LP_phase_1=time.time()-time_LP_phase_1
        model.write("Output_"+the_filename+"_phase1.attr") 

        obj_1=model.ObjVal
        for v in model.getVars():
            if v.ub<gp.GRB.INFINITY:
                v.ub = gp.GRB.INFINITY
        model.update()
        #epsilon=.1
        ##for var in model.getVars():
        #    noise = abs(random.uniform(0, epsilon))  # new random value for each var
         #   print('noise')
         #   print(noise)
         #   var.Obj += noise  # update objective coefficient
        model.update()
        print('Running Phase Two without Reset')
        print('Running Phase Two without Reset')
        print('Running Phase Two without Reset')
        print('Running Phase Two without Reset')
        print('Running Phase Two without Reset')
        print('Running Phase Two without Reset')
        #model.setParam("Method", 0)
        model.write("INPUT_"+the_filename+"_phase2.mps") 
        time_LP_phase_2=time.time()

        model.optimize()
        time_LP_phase_2=time.time()-time_LP_phase_2

        model.write("Output_"+the_filename+"_phase2.attr") 

        obj_2=model.ObjVal
        print('Running Phase Two WITH Reset')
        print('Running Phase Two WITH Reset')
        print('Running Phase Two WITH Reset')
        print('Running Phase Two WITH Reset')
        print('Running Phase Two WITH Reset')
        model.reset()
        model.setParam("Method", -1)

        model.write("INPUT_"+the_filename+"_phase2_with_reset.mps") 
        time_LP_phase_2_with_reset=time.time()

        model.optimize()
        time_LP_phase_2_with_reset=time.time()-time_LP_phase_2_with_reset
        model.write("Output_"+the_filename+"_phase2_with_reset.attr") 

        obj_2a=model.ObjVal

        print('time_LP_phase_1,obj_1')
        print([time_LP_phase_1,obj_1])
        print('[time_LP_phase_2,obj_2]')
        print([time_LP_phase_2,obj_2])
        print('time_LP_phase_2_with_reset,obj2a')
        print([time_LP_phase_2_with_reset,obj_2a])