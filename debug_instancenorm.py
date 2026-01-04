
import torch
import torch.nn as nn
import numpy as np

N, D = 32, 512
x = torch.randn(N, D)

bn = nn.InstanceNorm1d(D)
# We need to capture the warning or error if any, but let's check values.
try:
    out = bn(x)
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")
    if torch.allclose(out, torch.zeros_like(out), atol=1e-5):
        print("Output is all zeros!")
    else:
        print("Output is NOT all zeros.")
        print(f"Sample output: {out[0, :5]}")
except Exception as e:
    print(f"Error: {e}")

# Check LayerNorm used in Affine-Norm
ln = nn.LayerNorm(D)
out_ln = ln(x)
print(f"LayerNorm mean: {out_ln.mean():.4f}, std: {out_ln.std():.4f}")
print(f"Sample LayerNorm output: {out_ln[0, :5]}")
