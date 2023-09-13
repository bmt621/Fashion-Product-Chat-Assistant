class self_attend_configs():
    max_blocksize: int = 512
    hidden_dim: int = 768
    nhead: int  = 8
    use_flash_attn: bool = True
    dropout: float = 0.1
    bias : bool = True


class TransformerEncoderConfigs(self_attend_configs):
    n_encoder_layer: int = 5
    vocab_size: int = None
    embed_dim: int = 768
    sinusoid: bool = False
    norm_first: bool = True
    d_ff: int = 2048

class TransformerDecoderConfigs(self_attend_configs):
    n_decoder_layer: int = 5
    vocab_size: int = None
    embed_dim: int = 768
    sinusoid: bool  = False
    norm_first: bool = True

class EncoderDecoderConfigs(self_attend_configs):
    n_encoder_layer: int = 5
    n_decoder_layer: int = 5
    embed_dim: int = 768
    vocab_size: int =  None
    sinusoid: bool = False
    norm_first: bool = True