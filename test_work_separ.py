from class_new_valid import complete_separater_end_to_end 
import pickle

loaded_object=[]

print('staritng load ')
with open("USE_my_object.pkl", "rb") as f:
    loaded_object = pickle.load(f)

print('done load ')
#input('paused')
my_adder=complete_separater_end_to_end(loaded_object)
