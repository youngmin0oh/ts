
import torch
import sys
import os
sys.path.append('/home/youngmin0.oh/github/ts/Adapter-X+Y')
from model import Sundial

def inspect_sundial():
    print("Loading Sundial model...")
    class Config:
        def __init__(self):
            self.pred_len = 96
            self.seq_len = 512
            self.hf_model_id = 'thuml/sundial-base-128m' # Force correct ID
            
    configs = Config()
    wrapper = Sundial.Model(configs)
    model = wrapper.model
    
    print("\nModel Config:")
    print(model.config)
    
    print("\nModel Architecture:")
    print(model)
    
    # create dummy input
    x = torch.randn(1, 512, 1) # [B, L, C] -> Wrapper reshapes to [B*C, L] => [1, 512]
    # wrapper passes this to model.generate
    
    print("\nInspecting embed_layer...")
    embed_layer = model.model.embed_layer
    print(embed_layer)
    
    # Try [1, 512]
    try:
        x = torch.randn(1, 512).to(model.device)
        print("Testing embed_layer([1, 512])...")
        emb = embed_layer(x)
        print("Embed success! Shape:", emb.shape)
    except Exception as e:
        print("embed_layer([1, 512]) failed:", e)

    # Try [1, 512, 1]
    try:
        x = torch.randn(1, 512, 1).to(model.device)
        print("Testing embed_layer([1, 512, 1])...")
        emb = embed_layer(x)
        print("Embed success! Shape:", emb.shape)
    except Exception as e:
        print("embed_layer([1, 512, 1]) failed:", e)

    # Try [1, 512, 2]
    try:
        x = torch.randn(1, 512, 2).to(model.device)
        print("Testing embed_layer([1, 512, 2])...")
        emb = embed_layer(x)
    except Exception as e:
        print("embed_layer([1, 512, 2]) failed:", e)

    print("\nAttempting generate with inputs_embeds (Bypassing Mixin)...")
    try:
        from transformers.generation.utils import GenerationMixin
        x = torch.randn(1, 512).to(model.device)
        inputs_embeds = embed_layer(x)
        print("Embeddings shape:", inputs_embeds.shape)
        
        # Bypass custom generate method
        out = GenerationMixin.generate(model, inputs_embeds=inputs_embeds, max_new_tokens=96, use_cache=False)
        print("Bypassed Mixin SUCCESS!")
        print("Output shape:", out.shape)
    except Exception as e:
        print("Bypassed Mixin FAILED:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_sundial()
