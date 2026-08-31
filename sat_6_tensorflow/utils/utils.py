import pickle
import sys
import pandas as pd

    
def save_object(filename, object):
    with open(filename, 'wb') as outp:
        pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def get_model_output_filename (topology,quant):
    return f"./pytorch_models/sat6-cnn-t{topology}w{quant}.pt"

def get_hardware_config_name (topology,quant,target_fps):
    return f"t{topology}w{quant}_{target_fps}fps"

def save_csv_table(results,csv_pathname):
    df = pd.DataFrame(results)
    print(df.to_string(header=None, index=False))
    df.to_csv(csv_pathname) 
    print(f"succesfully saved at {csv_pathname}")

