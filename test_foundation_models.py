
import torch
import sys
import os

# Add the project root to path so we can import models
sys.path.append('/home/youngmin0.oh/github/ts/Adapter-X+Y')

from model import Sundial, TTM

class Config:
    def __init__(self):
        self.pred_len = 96
        self.seq_len = 512 # TTM usually likes 512 context
        # For Sundial, context might be fixed or flexible
        self.model_id = 'thuml/sundial-base-128m' # Default for Sundial

def test_sundial():
    print("Testing Sundial...")
    configs = Config()
    try:
        model = Sundial.Model(configs)
        print("Sundial Model instantiated.")
        
        # Batch=2, Seq=512, Channel=3
        x = torch.randn(2, 512, 3) 
        output = model(x, None, None, None)
        print("Sundial Forward Pass Successful.")
        print("Output shape:", output.shape) # Expect [2, 96, 3] usually
    except Exception as e:
        print("Sundial Test Failed:", e)

def test_ttm():
    print("\nTesting TTM...")
    configs = Config()
    configs.model_id = 'ibm/ttm-research-r2' # Using a known public model ID just to be safe or try specific one
    # Note: TTM models often cover specific context lengths. 
    # If we use a generic ID, it might load a specific config.
    
    try:
        model = TTM.Model(configs)
        print("TTM Model instantiated.")
        
        # Batch=2, Seq=512, Channel=3
        x = torch.randn(2, 512, 3)
        output = model(x, None, None, None)
        print("TTM Forward Pass Successful.")
        print("Output shape:", output.shape)
    except Exception as e:
        print("TTM Test Failed:", e)

if __name__ == "__main__":
    test_sundial()
    test_ttm()
