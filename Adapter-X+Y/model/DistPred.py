
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Minusformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from utils.tools import standard_scaler

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2402.02332
    Implementation of DistPred (Minusformer architecture).
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        
        # bins is required for DistPred (distribution estimation)
        # Default to 100 if not specified in configs
        self.bins = getattr(configs, 'bins', 100)
        
        self.embed = nn.Linear(configs.seq_len, configs.d_model)
        self.backbone = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads) if configs.attn else None,
                    configs.d_model,
                    self.pred_len, # d_block corresponds to pred_len in Minusformer
                    configs.d_ff,
                    dropout=configs.dropout,
                    gate = configs.gate
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.out_proj = nn.Linear(self.pred_len, configs.pred_len * self.bins)
        
        print('DistPred (Minusformer architecture) initialized...')


    def forward(self, x, x_mark, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input Length, Variate]
        # x_mark: [Batch, Input Length, Features]
        
        x = x.permute(0,2,1) # [Batch, Variate, Input Length]
        
        scaler = standard_scaler(x)
        x = scaler.transform(x)
        
        if x_mark is not None:
             # In Minusformer, x_mark is concatenated. 
             # x: [B, V, L], x_mark: [B, L, D] -> permute -> [B, D, L]
             # Cat dim=1 -> [B, V+D, L]
             # Embed maps L -> d_model.
             # So result is [B, V+D, d_model]
            x_emb = self.embed(torch.cat((x, x_mark.permute(0,2,1)),1))  
        else:
            x_emb = self.embed(x)
            
        output = self.backbone(x_emb) 
        
        output = self.out_proj(output)
        output = scaler.inverted(output[:, :x.size(1), :])
        
        # output shape: [Batch, Variate, pred_len * bins]
        # Reshape to [Batch, Variate, pred_len, bins]
        # And transpose to match expected format?
        # Minusformer.py: return output.reshape(x.size(0), x.size(1), self.pred_len, -1)
        # -> [Batch, Variate, PredLen, Bins]
        
        return output.reshape(x.size(0), x.size(1), self.pred_len, -1)
