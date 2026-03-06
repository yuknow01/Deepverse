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

class ChannelEmbedding(nn.Module):
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

class ProwiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = LayerNormalization(d_model=D_MODEL)

    def forward(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return self.norm(x + self.dropout(output))
    
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.ffn = ProwiseFeedForwardNet()
        self.norm = LayerNormalization(D_MODEL)
    
    def forward(self, enc_inputs):
        attn_output, attn = self.self_attn(enc_inputs, enc_inputs, enc_inputs)
        attn_output = self.norm(attn_output)
        enc_outputs = self.ffn(attn_output)
        
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.norm1 = LayerNormalization(D_MODEL)
        self.norm2 = LayerNormalization(D_MODEL)
        self.ffn = ProwiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs):
        dec_inputs, self_attn = self.self_attn(dec_inputs, dec_inputs, dec_inputs)
        dec_inputs = self.norm1(dec_inputs)

        dec_inputs, cross_attn = self.cross_attn(dec_inputs, enc_outputs, enc_outputs)
        dec_inputs = self.norm2(dec_inputs)
        
        dec_inputs = self.ffn(dec_inputs)

        return dec_inputs, cross_attn

class CSIFormer(nn.Module):
    """
    Transforemr model Encoder-Decoder 구조로 구현.
    """
    def __init__(
            self, 
            element_length=ELEMENT_LENGTH,
            d_model=D_MODEL,
            max_len=MAX_LEN,
            n_encoder_layers=N_LAYERS,
            n_decoder_layers=N_LAYERS,
            decoder_input_mode="last",  # "last" | "zeros" | "learnable"  , dec_in 없을 때 미래 토큰 개수
            t_dec=1
    ):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.max_len = max_len
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.decoder_input_mode = decoder_input_mode
        self.t_dec = t_dec

        self.enc_embed = ChannelEmbedding(element_length, d_model, max_len)
        self.dec_embed = ChannelEmbedding(element_length, d_model, max_len)

        self.enc_layers = nn.ModuleList([EncoderLayer() for _ in range(n_encoder_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer() for _ in range(n_decoder_layers)])

        self.pred_head = nn.Linear(d_model, element_length)

         # learnable query 모드용 (dec 토큰을 값 없이 “자리”로만 둠)
        if decoder_input_mode == "learnable":
            self.learnable_query = nn.Parameter(torch.randn(1, t_dec, d_model))

    def forward(self, enc_in, dec_in, return_hidden=False):
        # 1) Encoder
        enc_x = self.enc_embed(enc_in)  # (B,Tenc,d)
        for layer in self.enc_layers:
            enc_x, _ = layer(enc_x)
        enc_mem = enc_x  # memory

        dec_x = self.dec_embed(dec_in)

        for layer in self.dec_layers:
            dec_x, _ = layer(dec_x, enc_mem)

        # 4) Predict
        yhat = self.pred_head(dec_x)  # (B,Tdec,L)

        if return_hidden:
            return yhat, enc_mem, dec_x
        return yhat