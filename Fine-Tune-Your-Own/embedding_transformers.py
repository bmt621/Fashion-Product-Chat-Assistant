import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from torch.autograd import Variable
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias,eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias,eps=self.eps)
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_proj1    = nn.Linear(config.hidden_dim, config.d_ff, bias=config.bias)
        self.dropout1 = nn.Dropout(config.dropout)
        self.gelu1    = nn.GELU()
        self.c_proj2  = nn.Linear(config.d_ff, config.hidden_dim, bias=config.bias)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_proj1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.c_proj2(x)
        x = self.dropout2(x)
        return x
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout: float = 0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
    

class SelfAttend(nn.Module):
    def __init__(self,configs):
        super(SelfAttend,self).__init__()

        self.configs = configs

        self.query = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)
        self.key = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)
        self.value  = nn.Linear(self.configs.hidden_dim,self.configs.hidden_dim)

        self.proj = nn.Linear(self.configs.hidden_dim, self.configs.hidden_dim,bias=self.configs.bias)
        self.proj_dropout = nn.Dropout(self.configs.dropout)
        self.attn_dropout = nn.Dropout(0.1)
        

        self.use_flash_attn = configs.use_flash_attn
        self.give_info = False

    def get_qkv(self,xq,xk,xv):
        assert xq.shape[-1] == xk.shape[-1] == xv.shape[-1], self.logger.info('shape of query, key and value embedding dimension must be thesame')

        B, qT, hd = xq.shape
        _, kT, _ = xk.shape
        _, vT, _ = xv.shape

        assert kT == vT, self.logger.info('query length and values length should be the same')
        
        q = self.query(xq).view(B, qT, self.configs.nhead, -1).permute(0, 2, 1, 3)
        k = self.key(xk).view(B, kT, self.configs.nhead, -1).permute(0, 2, 1, 3)
        v = self.value(xv).view(B, vT, self.configs.nhead, -1).permute(0, 2, 1, 3)

        return q, k, v
    

    def forward(self,q, k, v,padding_mask = None, is_causal = False):
        
        if is_causal and not self.configs.use_flash_attn:
            self.register_buffer('attn_mask',torch.tril(torch.ones(1, 1, self.configs.max_blocksize,self.configs.max_blocksize,dtype=bool)))

        if self.configs.use_flash_attn:
            
            if is_causal:
                with torch.backends.cuda.sdp_kernel(enable_math=True): # use the most efficient implementation fused kernel
                    output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            else:
                with torch.backends.cuda.sdp_kernel(enable_math=True):
                    output = F.scaled_dot_product_attention(q, k, v,attn_mask = None, is_causal=False)

        else:
            
            scaler = (1.0 / math.sqrt(k.shape[-1]))
            attn_weight = (q @ k.transpose(2,3)) * scaler

            if is_causal:
                masked_weights = attn_weight.masked_fill(self.attn_mask[:,:,:q.shape[2],:q.shape[2]]==0, float('-inf'))

                if padding_mask is not None:
                    masked_weights.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

                norm_weights = F.softmax(masked_weights,dim=-1)
                norm_weights = self.attn_dropout(norm_weights)

                output = norm_weights @ v # (B, nh, T, hs)

            else:
                
                if padding_mask is not None:
                    attn_weight.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

                norm_weights = F.softmax(attn_weight,dim=-1)

                output = norm_weights @ v # (B, nh, T, hs)
                
            
        output = output.permute(0, 2, 1, 3).flatten(start_dim=2)    
        output = self.proj_dropout(self.proj(output))

        return output

        
class EncoderBlock(nn.Module):

    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.hidden_dim, eps=1e-5)
        self.attn = SelfAttend(config)

        self.ln_2 = nn.LayerNorm(config.hidden_dim, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x,src_padding_mask = None):

        if self.config.norm_first:

            x = self.ln_1(x)
            q, k, v = self.attn.get_qkv(x,x,x)

            x = x + self.attn(q, k, v, src_padding_mask)
            x = x + self.mlp(self.ln_2(x))
        else:
            q, k, v = self.attn.get_qkv(x,x,x)
            x = self.ln_1(x + self.attn(q, k, v, src_padding_mask))
            x = self.ln_2(x + self.mlp(x))

        return x
    
class DecoderBlock(nn.Module):

    def __init__(self,config):
        super(DecoderBlock, self).__init__()

        self.ln_1 = LayerNorm(config.hidden_dim, bias=config.bias)
        self.attn = SelfAttend(config)

        self.ln_2 = LayerNorm(config.hidden_dim, bias=config.bias)
        self.mlp = MLP(config)
        self.config = config

        

    def forward(self, x, mem, tgt_padding_mask = None):

        if self.config.norm_first:

            x = self.ln_1(x)
            q, k, v = self.attn.get_qkv(x, x, x)

            x = x + self.attn(q,k,v, is_causal=True)  # Masked Self-Attention
            q, k, v = self.attn.get_qkv(x, mem, mem)

            x = x + self.attn(q,k,v,is_causal=False, padding_mask = tgt_padding_mask) # Encoder-Decoder Attention
            x = x + self.mlp(self.ln_2(x))
        else:
            q, k, v = self.attn.get_qkv(x,x,x)
            x = x + self.attn(q,k,v, is_causal=True) # Masked Self-Attention

            q, k, v = self.attn.get_qkv(x, mem, mem) 
            x = x + self.attn(q, k, v, is_causal=False, padding_mask = tgt_padding_mask) # Encoder-Decoder Attention
            x = self.ln_2(x + self.mlp(x))

        return x
    
class TransformerEncoder(nn.Module):

    def __init__(self,configs):
        super(TransformerEncoder,self).__init__()
        self.configs = configs
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  # Create a logger instance
        
        self.Encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(configs.vocab_size,configs.embed_dim),
            wpe = nn.Embedding(configs.max_blocksize,configs.embed_dim) if not configs.sinusoid else PositionalEncoding(d_model = configs.embed_dim, max_len = configs.max_blocksize),
            dropout = nn.Dropout(configs.dropout),
            h_layer = nn.ModuleList(
            [EncoderBlock(configs) for _ in range(configs.n_encoder_layer)]
            ),
            ln = LayerNorm(configs.embed_dim,configs.bias),
        ))
        
        if self.configs.use_flash_attn:
            self.logger.info("This implementation of flash attention does not support src_key_padding_mask or tgt_key_padding_mask")
            self.give_info = True
    
    def forward(self,idx,src_padding_mask=None):

        device = idx.device
        b, t = idx.shape

        assert t<self.configs.max_blocksize, 'length on input token {} should not exceed the max_token_length {}'.format(t,self.configs.block_size)
        
        pos = torch.arange(0, t, dtype=torch.long,device=device)
        tok_emb = self.Encoder.wte(idx)
        
        if not self.configs.sinusoid:
            pos_emb = self.Encoder.wpe(pos)
            x = self.Encoder.dropout(tok_emb + pos_emb)
        else:
            x = self.Encoder.wpe(tok_emb)
        
        for block in self.Encoder.h_layer:
            x = block(x, src_padding_mask)
        
        x = self.Encoder.ln(x)

        return x
