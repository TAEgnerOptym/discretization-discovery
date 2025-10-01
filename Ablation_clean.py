from call_and_run_code import call_and_run_code
import os


#USER PARAMS BELOW


#option_use="normal"
#option_use="cuts_off_graphs_on"
#option_use="graphsOffCutsOn"
#option_use="no_cuts_or_graphs"
option_use="normal"
#no_cuts_or_graphs,

option_use="cuts_off_graphs_on"
dataSetUse="R2"
num_cust_use=100


#USER PARAMS ABOVE



params_prempt=dict()
params_prempt['Actions_of_given_sol']=[]
params_prempt['num_cust_use']=num_cust_use

SOLOMON_OBJ = {
    # RC
    ("RC101", 25): 461.1, ("RC101", 50): 944.0, ("RC101", 100): 1619.8,
    ("RC102", 25): 351.8, ("RC102", 50): 822.5, ("RC102", 100): 1457.4,
    ("RC103", 25): 332.8, ("RC103", 50): 710.9, ("RC103", 100): 1258.0,
    ("RC104", 25): 306.6, ("RC104", 50): 545.8, ("RC104", 100): 1132.3,
    ("RC105", 25): 411.3, ("RC105", 50): 855.3, ("RC105", 100): 1513.7,
    ("RC106", 25): 345.5, ("RC106", 50): 723.2, ("RC106", 100): 1372.7,
    ("RC107", 25): 298.3, ("RC107", 50): 642.7, ("RC107", 100): 1207.8,
    ("RC108", 25): 294.5, ("RC108", 50): 598.1, ("RC108", 100): 1114.2,

    ("RC201", 25): 360.2, ("RC201", 50): 684.8, ("RC201", 100): 1261.8,
    ("RC202", 25): 338.0, ("RC202", 50): 613.6, ("RC202", 100): 1092.3,
    ("RC203", 25): 326.9, ("RC203", 50): 555.3, ("RC203", 100): 923.7,
    ("RC204", 25): 299.7, ("RC204", 50): 444.2, ("RC204", 100): 999999999,
    ("RC205", 25): 338.0, ("RC205", 50): 630.2, ("RC205", 100): 1154.0,
    ("RC206", 25): 324.0, ("RC206", 50): 610.0, ("RC206", 100): 1051.1,
    ("RC207", 25): 298.3, ("RC207", 50): 558.6, ("RC207", 100): 962.9,
    ("RC208", 25): 269.1, ("RC208", 50): 476.7, ("RC208", 100): 999999999,

    # R
    ("R101", 25): 617.1, ("R101", 50): 1044.0, ("R101", 100): 1637.7,
    ("R102", 25): 547.1, ("R102", 50): 909.0, ("R102", 100): 1466.6,
    ("R103", 25): 454.6, ("R103", 50): 772.9, ("R103", 100): 1208.7,
    ("R104", 25): 416.9, ("R104", 50): 625.4, ("R104", 100): 971.5,
    ("R105", 25): 530.5, ("R105", 50): 899.3, ("R105", 100): 1355.3,
    ("R106", 25): 465.4, ("R106", 50): 793.0, ("R106", 100): 1234.6,
    ("R107", 25): 424.3, ("R107", 50): 711.1, ("R107", 100): 1064.6,
    ("R108", 25): 397.3, ("R108", 50): 617.7, ("R108", 100): 932.1,
    ("R109", 25): 441.3, ("R109", 50): 786.8, ("R109", 100): 1146.9,
    ("R110", 25): 444.1, ("R110", 50): 697.0, ("R110", 100): 1068.0,
    ("R111", 25): 428.8, ("R111", 50): 707.2, ("R111", 100): 1048.7,
    ("R112", 25): 393.0, ("R112", 50): 630.2, ("R112", 100): 948.6,

    ("R201", 25): 463.3, ("R201", 50): 791.9, ("R201", 100): 1143.2,
    ("R202", 25): 410.5, ("R202", 50): 698.5, ("R202", 100): 1029.6,
    ("R203", 25): 391.4, ("R203", 50): 605.3, ("R203", 100): 870.8,
    ("R204", 25): 355.0, ("R204", 50): 506.4, ("R204", 100): 999999999,
    ("R205", 25): 393.0, ("R205", 50): 690.1, ("R205", 100): 949.8,
    ("R206", 25): 374.4, ("R206", 50): 632.4, ("R206", 100): 875.9,
    ("R207", 25): 361.6, ("R207", 50): 575.5, ("R207", 100): 794.0,
    ("R208", 25): 328.2, ("R208", 50): 487.7, ("R208", 100): 999999999,
    ("R209", 25): 370.7, ("R209", 50): 600.6, ("R209", 100): 854.8,
    ("R210", 25): 404.6, ("R210", 50): 645.6, ("R210", 100): 900.5,
    ("R211", 25): 350.9, ("R211", 50): 535.5, ("R211", 100): 999999999,

    # C
    ("C101", 25): 191.3, ("C101", 50): 362.4, ("C101", 100): 827.3,
    ("C102", 25): 190.3, ("C102", 50): 361.4, ("C102", 100): 827.3,
    ("C103", 25): 190.3, ("C103", 50): 361.4, ("C103", 100): 826.3,
    ("C104", 25): 186.9, ("C104", 50): 358.0, ("C104", 100): 822.9,
    ("C105", 25): 191.3, ("C105", 50): 362.4, ("C105", 100): 827.3,
    ("C106", 25): 191.3, ("C106", 50): 362.4, ("C106", 100): 827.3,
    ("C107", 25): 191.3, ("C107", 50): 362.4, ("C107", 100): 827.3,
    ("C108", 25): 191.3, ("C108", 50): 362.4, ("C108", 100): 827.3,
    ("C109", 25): 191.3, ("C109", 50): 362.4, ("C109", 100): 827.3,

    ("C201", 25): 214.7, ("C201", 50): 360.2, ("C201", 100): 589.1,
    ("C202", 25): 214.7, ("C202", 50): 360.2, ("C202", 100): 589.1,
    ("C203", 25): 214.7, ("C203", 50): 359.8, ("C203", 100): 588.7,
    ("C204", 25): 213.1, ("C204", 50): 350.1, ("C204", 100): 588.1,
    ("C205", 25): 214.7, ("C205", 50): 359.8, ("C205", 100): 586.4,
    ("C206", 25): 214.7, ("C206", 50): 359.8, ("C206", 100): 586.0,
    ("C207", 25): 214.5, ("C207", 50): 359.6, ("C207", 100): 585.8,
    ("C208", 25): 214.5, ("C208", 50): 350.5, ("C208", 100): 585.8,
}

my_nums=["01","02","03","04","05","06","07","08","09","10","11","12"]
dataSetNums=dict()
dataSetNums['C1']=my_nums[0:9]
dataSetNums['C2']=my_nums[0:8]
dataSetNums['R1']=my_nums[0:12]
dataSetNums['R2']=my_nums[0:11]
dataSetNums['RC1']=my_nums[0:8]
dataSetNums['RC2']=my_nums[0:8]

if option_use=="cuts_off_graphs_on":
    out_fold_path_portion_ablation_name="cuts_off_graphs_on/"
    params_prempt['use_ineq']=False

if option_use=="normal":
    out_fold_path_portion_ablation_name="normal/"
if option_use=="graphs_offCutsOn":
    out_fold_path_portion_ablation_name="graphsOffCutsOn/"
    params_prempt['use_NG_graph']=False
    params_prempt['use_time_graph']=False
    params_prempt['use_dem_graph']=False
    params_prempt['use_ineq']=True
if option_use=="no_cuts_or_graphs":
    out_fold_path_portion_ablation_name="no_cuts_or_graphs/"
    params_prempt['use_NG_graph']=False
    params_prempt['use_time_graph']=False
    params_prempt['use_dem_graph']=False
    params_prempt['use_ineq']=False
if option_use=="no_ub_use_remove":
    out_fold_path_portion_ablation_name="no_ub_use_remove/"

    params_prempt['ub_use_remove']=99999999999

my_json_input_path="../mid_jnk"
data_fold='data/'
output_file_fold='../WillRezAbl_split_based_on/'+dataSetUse+'_numCust_'+str(num_cust_use)+'/'+option_use+'/'
param_file_path='params_WW/params_ablation/'+dataSetUse+'.json'
for k in range(0,len(dataSetNums[dataSetUse])):

    file_name="jy_"+str(dataSetUse)+str(dataSetNums[dataSetUse][k])
    input_file_path="data/"+file_name
    
    output_file_path=output_file_fold+file_name
    #ram_file_path=param_file_fold+file_name
    input_file_path=data_fold+file_name.lower()+'.txt'
    params_prempt['Actions_of_given_sol']=[]
    print(dataSetNums[dataSetUse][k])
    print(dataSetNums[dataSetUse])
    if option_use!='no_ub_use_remove':
        key_2_objective=dataSetUse+str(dataSetNums[dataSetUse][k])
        params_prempt['ub_use_remove']=SOLOMON_OBJ[(key_2_objective,num_cust_use)]
    
    os.makedirs(output_file_fold, exist_ok=True)
    
    
    if not os.path.exists(output_file_path):
    #if   not os.path.exists(output_file_path):
        print("input_file_path")
        print(input_file_path)
        print("param_file_path")
        print(param_file_path)
        print('params_prempt[ub_use_remove]')
        print(params_prempt['ub_use_remove'])
        print('params_prempt')
        print(params_prempt)
        print("_----")
        #input('---')
        call_and_run_code(input_file_path, param_file_path, my_json_input_path, output_file_path,params_prempt)
    else:
        print(f"Output file {output_file_path} already exists. Skipping call.")