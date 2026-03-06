import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ELEMENT_LENGTH = 64
D_MODEL = 128
MAX_LEN = 129
N_LAYERS = 12
N_HEADS = 8
D_FF = D_MODEL * 4
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS
DROPOUT = 0.1

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class Embedding(nn.Module):
    def __init__(self, element_length, d_model, max_len):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x[:, :, 0])
        tok_emb = self.proj(x.float())  
        embedding = tok_emb + self.pos_embed(pos)
        return self.norm(embedding)
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS)
        self.out = nn.Linear(N_HEADS * D_V, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0) 
        q_s = self.W_Q(Q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)
       
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, N_HEADS * D_V)
        output = self.out(context)
        
        return residual + self.dropout(output), attn
    
class PowiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = LayerNormalization(D_MODEL)
    
    def forward(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return self.norm(x + self.dropout(output))

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.ffn = PowiseFeedForwardNet()
        self.norm = LayerNormalization(D_MODEL)

    def forward(self, enc_intput):
        enc_intput, attn = self.self_attn(enc_intput, enc_intput, enc_intput)
        enc_intput = self.norm(enc_intput)
        enc_intput = self.ffn(enc_intput)
        return enc_intput, attn
       
class CPPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(ELEMENT_LENGTH, D_MODEL, MAX_LEN)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])
        self.ffn = PowiseFeedForwardNet()
    
    def forward(self, input_ids):
        """
        전처리를 진행한 후 코드 진행 complex 된 channel concat하기
        """
        output = self.embedding(input_ids)
        for layer in self.layers:
            output, _ = layer(output)
        output = self.ffn(output)

        return output


           