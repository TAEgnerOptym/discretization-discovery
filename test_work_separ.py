from class_new_valid import complete_separater_end_to_end 
import pickle

loaded_object=[]

print('staritng load ')
#with open("../ALL_JSON_BIG/GOOD_my_object.pkl", "rb") as f:
my_file_name="GOOD_my_object.pkl"
my_file_name="R104_iter_1.pkl"
my_file_name="LAST_112_my_object.pkl"
with open(my_file_name, "rb") as f:
    loaded_object = pickle.load(f)

print('done load ')
#input('paused')
num_LA_cutting_plane=8
max_SRI_Divisor=2
max_SRI_SET_SIZE=3
use_custom_ng=True
my_adder=complete_separater_end_to_end(loaded_object,use_custom_ng,num_LA_cutting_plane,max_SRI_Divisor,max_SRI_SET_SIZE)
my_adder.update_given_solution()