
import torch
import sys
import inspect
sys.path.append('/home/youngmin0.oh/github/ts/Adapter-X+Y')
from model import Sundial

class Config:
    def __init__(self):
        self.pred_len = 96
        self.seq_len = 512
        self.model_id = 'thuml/sundial-base-128m'

def inspect_model():
    configs = Config()
    try:
        wrapper = Sundial.Model(configs)
        model = wrapper.model
        print("Model Class:", type(model))
        print("\nForward Signature:")
        print(inspect.signature(model.forward))
        
        print("\nGenerate Signature:")
        print(inspect.signature(model.generate))
        
        # Check if there is a 'tokenize' or 'patch' method
        print("\nDir Model:")
        print(dir(model))
        
    except Exception as e:
        print("Inspection Failed:", e)

if __name__ == "__main__":
    inspect_model()
