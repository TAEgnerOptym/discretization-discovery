
import json

from grab_default_params import grab_default_params
def grab_params(param_path):


	f = open(param_path)
	
	user_params = json.load(f)

	my_params= grab_default_params()
	
	for i in user_params:
		my_params[i]=user_params[i]

	return my_params