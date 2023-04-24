import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def save_json(json_dict, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, cls=NumpyEncoder)

def save_npy(np_array, np_path):
    np.save(np_path, np_array)

def load_json(json_path):
    with open(json_path, "r") as f:
        json_dict = json.load(f)

    return json_dict

def load_npy(np_path):
    np_array = np.load(np_path)
    return np_array