import torch
import torch.nn as nn
from .activation import activation_layer


class DNN(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int=2, 
                 hidden_dims: list[int]=[128, 64], 
                 activation: str='relu', 
                 dropout: float=0.0, 
                 dice_dim: int=2,
                 sign_vocab_size: int=10000, 
                 sign_embedding_dim: int=16, 
                 sign_seq_len: int=50, 
                 use_sign_embedding: bool=False, 
                 sign_pooling: str='mean'):
        
        super().__init__()
        self.use_sign_embedding = use_sign_embedding
        if use_sign_embedding:
            assert sign_vocab_size is not None and sign_embedding_dim is not None and sign_seq_len is not None
            self.sign_embedding = nn.Embedding(sign_vocab_size, sign_embedding_dim, padding_idx=0)
            self.sign_seq_len = sign_seq_len
            self.sign_pooling = sign_pooling
            total_input_dim = input_dim + sign_embedding_dim
        else:
            total_input_dim = input_dim
        layers = []
        prev_dim = total_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_layer(activation, h_dim, dice_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sign_id_seq=None):
        # x: [batch_size, input_dim], sign_id_seq: [batch_size, seq_len]
        if self.use_sign_embedding and sign_id_seq is not None:
            emb = self.sign_embedding(sign_id_seq)  # [batch, seq_len, emb_dim]
            if self.sign_pooling == 'mean':
                emb_pooled = emb.mean(dim=1)
            elif self.sign_pooling == 'max':
                emb_pooled, _ = emb.max(dim=1)
            else:
                emb_pooled = emb.mean(dim=1)
            x = torch.cat([x, emb_pooled], dim=1)
        return self.net(x)

if __name__ == '__main__':
    model = DNN(input_dim=1024, output_dim=2, hidden_dims=[128, 64, 32], activation='dice', dropout=0.1, dice_dim=2)
    x = torch.randn(10240, 1024)
    out = model(x)
    print(out.shape)