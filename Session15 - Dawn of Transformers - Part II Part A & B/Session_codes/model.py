from torch.nn.modules.activation import MultiheadAttention
import torch
import torch.nn as nn
import math


## coding our own LAYER NORMALISZATION CODE as the inbuilt one doesnt allow bias = false
class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6)-> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))    ## alpha and beta are learnable parameters

    def forward(self, x):
        # x: [batch, seq_length , hidden_size]
        mean = x.mean(-1, keepdim = True) # [batch, seq, 1]
        std = x.std(-1, keepdim = True) # [batch , seq , 1]
        #keep the dimension for broadcasting , if (keepdim = False) - the last dimension will not be there - [batch , seqdim]
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


## FEED FORWARD NETWORK - using squeeze and expand method
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout:float)-> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model , d_ff) ## w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        ## [batch, seq_length, d_model] -> [batch, seq_length, d_ff] -> [batch, seq_length, d_model]
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

## for converting inputs to dimensional embedding prepared to go in encoder or decoder.
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int)-> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        # multiply by sqrt(d_model) to scale the embedding according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float )-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape [seq_len]
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) ## [seq_len , 1]
        # create a vector of shape [d_model]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model]
        # apply sine to even indices
        pe[ : , 0::2] = torch.sin(position * div_term) # sin(position * (10000**(2i/d_model))
        # apply cosine to all odd indices
        pe[ : , 1::2] = torch.cos(position * div_term) # cos(position * (10000**(2i/d_model))
        # add a batch to positional encoding
        pe = pe.unsqueeze(0)
        ## register the positional encoding as BUFFER(non trainable)
        self.register_buffer('pe' , pe) ## saves the value of pe as "pe" even if the kernel gets closed, and this is not back_propagated


    def forward(self, x):
        # x = x + (self.pe[:, : x.shape[1] , :]).requires_grad_(False) # [batch, seq_len , d_model]
        x = x + (self.pe[:, : , :]).requires_grad_(False)
        # x.shape[1] gives the seq_length of a sentence.
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout:float)-> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) ## cant understand it currently


## MULTI HEAD ATTENTION part which we can use for BOTH ENCODER and DECODER
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float)-> None:
        super().__init__()
        self.d_model = d_model # embedding vector size
        self.h = h # Number of heads
        #make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # dimension of embedding seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias = False) #Wq
        self.w_k = nn.Linear(d_model, d_model, bias = False) #Wk
        self.w_v = nn.Linear(d_model, d_model, bias = False) #Wv
        self.w_o = nn.Linear(d_model, d_model, bias = False) #Wo
        ## Heads are not considered yet in the above code
        self.dropout = nn.Dropout(dropout)

    @staticmethod  # we can directly use call function without instantiating the multi head attention class by using: MultiHeadAtetntion.attention(...)
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1] # gives the last dimension which is the dimension_size of each head i.e. d_k
        # Just apply formula from the paper
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            ## write a very low value(indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        # applying softmax along the values of last dimension(could have been applied along any of last 2 dimensions, doesnt matter)
        attention_scores = attention_scores.softmax(dim = -1) # [batch, h, seq_length, seq_length]
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        ## (batch, h, seq_length, seq_length) -> (batch, h , seq_length, d_k)
        # return attention scores which can be used for visualisation
        return (attention_scores @ value) , attention_scores

    def forward(self, q, k , v , mask):
        query = self.w_q(q) ## [batch, seq_length, d_model] -> [batch, seq_length, d_model]
        key = self.w_k(k) # [batch, seq_length, d_model] -> [batch, seq_length, d_model]
        value = self.w_v(v) # [batch, seq_length, d_model] -> [batch, seq_length, d_model]

        # dividing it into h heads
        #[batch, seq_length, d_model] -> [batch, seq_length, h, d_k] -> [batch, h, seq_length, d_k]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # query.shape[1] = seq_length(), query.shape[0] = batch
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        #calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #combine all heads together
        # (batch, h, seq_length, d_k) -> (batch, seq_length, h , d_k) - > (batch , seq_length, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0] , -1, self.h * self.d_k)

        # multipply by wo
        # (batch , seq_length, d_model) -> (batch , seq_length, d_model)
        return self.w_o(x)

## a single encoder block
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock , dropout:float)-> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        ## as for an encoder key, query, value have same inputs
        x = self.residual_connection[1](x, self.feed_forward_block)
        # in encoder block, one can see 2 skip connections, one before and after the MHA and one before after the Feed forward layer.
        return x

## Actual encoder
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x ,x, x, tgt_mask))
        # initial masked multi head attention layer where encoder outputs are not used
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # cross attention layers where query is x, and key and value are from encoder blocks
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
        # final feed forward layer

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList)-> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layers in self.layers:
            x = layers( x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


## for converting the final enmbedding to the vocabulary space meaning which work is most likely to come
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x)-> None:
        # [batch, seq_length, d_model] -> [batch, seq_length, vocab_size]
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed : InputEmbeddings, tgt_embed : InputEmbeddings, src_pos : PositionalEncoding, tgt_pos : PositionalEncoding, projection_layer : ProjectionLayer  )-> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        #[batch, seq_length, d_model]
        src = self.src_embed(src)
        src = self.src_pos(src)
        encoder_output = self.encoder(src, src_mask)
        return encoder_output

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor )-> None:
        # [batch, seq_length, d_model]
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        # target - the thing we need to predict
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

    def project(self, x):
        # [batch, seq_length, vocab_size]
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_length: int, tgt_seq_length: int, d_model: int = 512, N:int=6, h:int=8, dropout:float = 0.1, d_ff:int=2048):
    # create embedding layer

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_length, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_length, dropout)

    # create encoder blocks
    encoder_blocks = []
    # N - no of encoder and decoder blocks
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)


    # create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout )
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialise the parameters(will work even if we dont do this)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer