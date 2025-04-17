
from convert_dict_keys_str_to_tuple import *

def clean_up_json_input_post_process(D):


    for h in D['allGraphNames']:
        D['hij2P'][h]=convert_dict_keys_str_to_tuple(D['hij2P'][h])

    D['deltaCon2Contrib']=convert_dict_keys_str_to_tuple(D['deltaCon2Contrib'])
    D['actionCon2Contrib']=convert_dict_keys_str_to_tuple(D['actionCon2Contrib'])
    #
    D['primIntegCon2Contrib']=convert_dict_keys_str_to_tuple(D['primIntegCon2Contrib'])
    D['actionIntegCon2Contrib']=convert_dict_keys_str_to_tuple(D['actionIntegCon2Contrib'])


    D['hij2P'][h]
    return D