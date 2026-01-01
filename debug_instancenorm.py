
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostProcessingNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, delta=0.1):
        super(PostProcessingNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.bn1 = torch.nn.InstanceNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.InstanceNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.delta = delta
        # print('PostProcessingNet delta:', self.delta)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Check shape before bn1
        # print(f"Shape before bn1: {x.shape}")
        
        # In the original code:
        x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return torch.tanh(x) * self.delta

def test_model():
    state_dim = 96 * 7
    hidden_dim = 512
    action_dim = 96 * 7
    batch_size = 32
    
    model = PostProcessingNet(state_dim, hidden_dim, action_dim)
    model.train() # Set to train mode
    
    x = torch.randn(batch_size, state_dim)
    
    try:
        out = model(x)
        print("Model forward pass successful!")
        print("Output shape:", out.shape)
        print("Output (first 5 elements):", out[0, :5])
        print("Is output all zeros?", torch.allclose(out, torch.zeros_like(out)))
        
        # Check gradients
        loss = out.sum()
        loss.backward()
        print("Backward pass successful!")
        
    except Exception as e:
        print("Model forward pass failed:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
