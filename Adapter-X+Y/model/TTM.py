
import torch
import torch.nn as nn
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # model_id default
        hf_model_id = getattr(configs, 'hf_model_id', None)
        if hf_model_id is None:
            config_id = getattr(configs, 'model_id', '')
            if '/' in config_id:
                hf_model_id = config_id
            else:
                hf_model_id = 'ibm/ttm-research-r2'
        
        model_id = hf_model_id
        # Or specific variant like 'ttm-512-96-r2'
        # We might need to map configs.seq_len/pred_len to specific TTM checkpoints
        # TTM usually has fixed context lengths (e.g. 512, 1024)
        # For reproduction, we typically use the closest matching or standard one.
        # Let's assume a default or allow strictly passing via configs.
        # For Table 1 reproduction, we usually follow the paper's setup.
        # Assuming 'ibm-granite/granite-timeseries-ttm-r2' is the base.
        
        try:
            self.model = TinyTimeMixerForPrediction.from_pretrained(model_id)
        except Exception as e:
            # Fallback or error if model_id is not found/valid
            # Maybe try a more specific one if needed
            print(f"Error loading TTM model {model_id}: {e}")
            # Potentially fallback to a local path or a default
            self.model = TinyTimeMixerForPrediction.from_pretrained('ibm/ttm-research-r2') # Example fallback

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channels]
        # TTM typically takes [Batch, Seq_Len, Channels]
        
        # Create dummy freq_token (zeros) of shape [Batch]
        # Assuming 0 is a valid frequency index (usually 'other' or similar)
        # We need it on the same device as x_enc
        batch_size = x_enc.shape[0]
        freq_token = torch.zeros(batch_size, device=x_enc.device, dtype=torch.long)
        
        outputs = self.model(past_values=x_enc, freq_token=freq_token)
        
        # outputs is typically a HF model output object.
        # prediction_logits or similar.
        # TTM output: prediction_outputs (Tensor) of shape (batch_size, prediction_length, num_input_channels)
        
        return outputs.prediction_outputs
