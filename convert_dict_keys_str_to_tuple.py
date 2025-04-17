
import numpy as np
import ast
import json


def convert_dict_keys_str_to_tuple(x_dict):
    """
    Converts the keys of dictionary x_dict from strings to tuples,
    assuming each key is a string representation of a tuple of two elements.
    
    Parameters:
        x_dict (dict): Dictionary whose keys are strings representing a two-element tuple.
        
    Returns:
        dict: A new dictionary where the keys are actual tuples.
    """
    new_dict = {}
    for key, value in x_dict.items():
        try:
            # Safely evaluate the string to its Python literal (expects tuple)
            tuple_key = ast.literal_eval(key)
            
            tup0a=str(tuple_key[0])
            tup1a=str(tuple_key[1])
            tuple_key=tuple([tup0a,tup1a])
            if isinstance(tuple_key, tuple) and len(tuple_key) == 2:
                new_dict[tuple_key] = value
            else:
                # Optionally, handle keys that don't convert properly.
                # Here, we simply copy them as is.
                new_dict[key] = value
        except Exception as error:
            # In case evaluation fails, keep the original key.
            new_dict[key] = value
    return new_dict
