
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationMixin
import transformers.cache_utils

# Monkey-patch DynamicCache for Sundial compatibility
if hasattr(transformers.cache_utils, 'DynamicCache'):
    if not hasattr(transformers.cache_utils.DynamicCache, 'seen_tokens'):
        @property
        def seen_tokens(self):
            return self.get_seq_length()
        transformers.cache_utils.DynamicCache.seen_tokens = seen_tokens
    
    if not hasattr(transformers.cache_utils.DynamicCache, 'get_max_length'):
        def get_max_length(self):
            return None # Assume unlimited or handle by logic
        transformers.cache_utils.DynamicCache.get_max_length = get_max_length

    if not hasattr(transformers.cache_utils.DynamicCache, 'get_usable_length'):
        def get_usable_length(self, *args, **kwargs):
             return self.get_seq_length()
        transformers.cache_utils.DynamicCache.get_usable_length = get_usable_length

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        # Load the model from Hugging Face
        # Using trust_remote_code=True as required by Sundial
        # We assume the model name is passed or we default to 'thuml/sundial-base-128m'
        # If configs has a specific field for model path, we use it, else default.
        hf_model_id = getattr(configs, 'hf_model_id', None)
        if hf_model_id is None:
            config_id = getattr(configs, 'model_id', '')
            if '/' in config_id:
                hf_model_id = config_id
            else:
                hf_model_id = 'thuml/sundial-base-128m'
        model_id = hf_model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        
        # Override prepare_inputs_for_generation to correctly handle inputs_embeds
        # and prevent the model from defaulting to input_ids (which causes shape mismatch/crash)
        def custom_prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
            model_inputs = {}
            if inputs_embeds is not None:
                model_inputs['inputs_embeds'] = inputs_embeds
                # Crucial: Do NOT pass input_ids if inputs_embeds is present.
                # This forces the underlying SundialModel to use inputs_embeds (assuming it has check for input_ids is None)
            else:
                model_inputs['input_ids'] = input_ids
                
            if past_key_values:
                 model_inputs['past_key_values'] = past_key_values
                 # Also handle cache length logic which caused crashes
                 if hasattr(past_key_values, 'get_seq_length'):
                     # Ensure model can access length
                     pass
            
            return model_inputs
            
        # Bind the custom method to the model instance (MethodType not strictly needed if we assign to instance dict or use types)
        # Easier: assign to the instance method
        import types
        self.model.prepare_inputs_for_generation = types.MethodType(custom_prepare_inputs_for_generation, self.model)

        # Patch _extract_past_from_model_output which is missing but called by ts_generation_mixin
        def _extract_past_from_model_output(self, outputs, *args, **kwargs):
             if hasattr(outputs, 'past_key_values'):
                 return outputs.past_key_values
             return None
        self.model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, self.model)
        
        # Freeze parameters for Foundation Model Adapter approach
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc shape: [Batch, Seq_Len, Channels]
        B, L, C = x_enc.shape
        
        # Sundial expects [Batch, Length] for univariate or we treat channels as batch
        # Reshape to [Batch * Channels, Length]
        x_flattened = x_enc.permute(0, 2, 1).reshape(B * C, L)
        
        # Calculate embeddings manually to correct shape mismatch (512 vs 32)
        # embed_layer transforms [B*C, 512] -> [B*C, 32, Hidden]
        # This is CRITICAL because Sundial's custom mixin assumes inputs length matches patch length,
        # but transformers uses raw input length if we pass inputs directly.
        all_embeddings = self.model.model.embed_layer(x_flattened)
        
        # Inference Chunking to avoid OOM with large channel counts (e.g. ELC 321 channels)
        # We process 'all_embeddings' in small batches.
        # Effective batch size is B*C. ELC: 1 * 321 = 321.
        mini_batch_size = 16 # Adjust based on GPU memory. 16 * 128M params + cache fits easily.
        
        num_samples = all_embeddings.shape[0]
        all_preds = []
        
        for i in range(0, num_samples, mini_batch_size):
            end_idx = min(i + mini_batch_size, num_samples)
            batch_embeds = all_embeddings[i:end_idx]
            
            # Generate predictions
            # Use GenerationMixin directly to bypass Sundial's mixin which requires 'inputs' (causing failure with inputs_embeds)
            with torch.no_grad(): # Ensure no gradients during generation loop to save memory (even more)
                # Note: If we needed gradients for Ada-X, this chunking would need to be differentiable (no torch.no_grad).
                # But we OOM'd on Ada-X anyway. 
                # For Ada-Y, we detach outputs anyway.
                # So torch.no_grad() is safe and recommended for inference efficiency.
                chunk_outputs = GenerationMixin.generate(
                    self.model,
                    inputs_embeds=batch_embeds,
                    max_new_tokens=self.pred_len,
                    min_new_tokens=self.pred_len, # Force full length
                    max_length=batch_embeds.shape[1] + self.pred_len + 10, # Explicitly allowing growth
                    num_samples=1,
                    use_cache=True # Enable caching
                )
                # print(f"DEBUG: Chunk input {batch_embeds.shape}, Output {chunk_outputs.shape}")

            # Extract predictions for this chunk
            history_len = batch_embeds.shape[1]
            if len(chunk_outputs.shape) == 2: # [Batch, Seq_Len]
                 # If generation returned only new tokens (length == pred_len)
                 if chunk_outputs.shape[1] == self.pred_len:
                     chunk_pred = chunk_outputs
                 else:
                     # Otherwise assume it includes input (dummy IDs) and slice
                     chunk_pred = chunk_outputs[:, history_len:]
            elif len(chunk_outputs.shape) == 3: # [Batch, Num_Samples, Seq_Len]
                 if chunk_outputs.shape[2] == self.pred_len:
                     chunk_pred = chunk_outputs[:, 0, :]
                 else:
                     chunk_pred = chunk_outputs[:, 0, history_len:]
            else:
                 chunk_pred = chunk_outputs[..., -self.pred_len:]
                 
            if chunk_pred.shape[-1] != self.pred_len:
                 chunk_pred = chunk_pred[..., -self.pred_len:]
                 
            all_preds.append(chunk_pred)
            
        # Concatenate chunks
        pred = torch.cat(all_preds, dim=0)

        # Reshape to [B, Pred_Len, C]
        # pred is currently [B*C, Pred_Len]
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, Pred_Len, C]
        
        return pred
