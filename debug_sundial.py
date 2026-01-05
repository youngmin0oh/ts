
import torch
import sys
import os

sys.path.append('/home/youngmin0.oh/github/ts/Adapter-X+Y')
from model import Sundial

class Config:
    def __init__(self, seq_len=512):
        self.pred_len = 96
        self.seq_len = seq_len
        self.model_id = 'thuml/sundial-base-128m'

def test_sundial(seq_len):
    print(f"Testing Sundial with seq_len={seq_len}...")
    configs = Config(seq_len)
    try:
        model = Sundial.Model(configs)
        # Batch=1, Seq=seq_len, Channel=1 (Univariate) - check if it works for univariate first
        x = torch.randn(1, seq_len, 1) 
        output = model(x, None, None, None)
        print(f"Success with seq_len={seq_len}. Output: {output.shape}")
    except Exception as e:
        print(f"Failed with seq_len={seq_len}: {e}")

if __name__ == "__main__":
    # Test typical values
    # The error was 512 vs 32.
    # Maybe it wants 32?
    test_sundial(32)
    test_sundial(512)
