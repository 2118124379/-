import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
USE_CUDA = torch.cuda.is_available()
def judge(x,s):
    for i in x:
        for j in i:
            for k in j:
                if math.isnan(k):
                    print(s)
                    exit(0)
def mysoftmax(x):
    mxsum = 0
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            sum=0
            for k in x[i][j]:
                sum+=math.exp(k)
            for k in range(x.size(2)):
                x[i][j][k]=math.exp(x[i][j][k])/sum
            mxsum = max(mxsum,sum)
    print(mxsum)
    return x
def judgemask(x,s):
    for i in x:
        for j in i:
            flag=False
            for k in j:
                flag = flag or k
            if not flag:
                print(s)
                exit(0)
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0, debug=False):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.debug = debug
    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))#k的后两维转置和q相乘，bmm：tensor的矩阵乘法
        if scale != None:
            attention = attention * scale#scale 缩放因子
        if attn_mask != None:
            if self.debug:
                judgemask(attn_mask,'eeet')
            # print(q.size(),k.size(),attention.size(),attn_mask.size())
            attention = attention.masked_fill_(attn_mask, -np.inf)#-inf做softmax会趋近0
        if self.debug:
            attention=mysoftmax(attention)
            print(attention.max())
        else:
            attention = self.softmax(attention)
        if self.debug:
            judge(attention,'2')
        attention = self.dropout(attention)
        if self.debug:
            judge(attention,'3')
        context = torch.bmm(attention, v)
        if self.debug:
            judge(context,'4')
        return context, attention
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0, debug=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.dot_product_attention = ScaledDotProductAttention(dropout,debug)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.debug = debug
    def forward(self, key, value, query, attn_mask=None):#(batch_size,seq_len,model_dim)
        # print('x',key.size())
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        # print(batch_size)
        key = self.linear_k(key)#(batch_size,seq_len,model_dim)
        value = self.linear_v(value)
        query = self.linear_q(query)
        key = key.view(batch_size * num_heads, -1, dim_per_head)#(batch_size * num_heads,seq_len,dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)
        # print(key.size())
        if attn_mask != None:
            # print('###',attn_mask.size())
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
            # print(attn_mask.size())
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention( query, key, value, scale, attn_mask)
        if self.debug:
            judge(context,'context')
        context = context.view(batch_size, -1, dim_per_head * num_heads)#(batch_size,seq_len,model_dim)
        output = self.linear_final(context)
        output = self.dropout(output)

        output = self.layer_norm(residual + output)#Add & Norm
        return output, attention
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):#ffn_dim:隐藏神经元个数
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    def forward(self, inputs):
        residual = inputs
        output = self.w2(F.relu(self.w1(inputs)))#两次线性变换+relu激活
        output = self.dropout(output)
        output = self.layer_norm(residual + output)#Add & Norm
        return output
def padding_mask(seq_k, seq_q):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(3).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    # print(type(pad_attn_mask))
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    if USE_CUDA:
        pad_attn_mask = pad_attn_mask.cuda()
    return pad_attn_mask
def sequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    mask = torch.from_numpy(mask).byte()
    if USE_CUDA:
        mask = mask.cuda()
    return mask# [batch_size, tgt_len, tgt_len]
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, pad):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model,padding_idx=pad)
        self.d_model = d_model
 
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])#论文里的公式
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.from_numpy(position_encoding)
        position_encoding = torch.cat((pad_row, position_encoding))#添加一个对pad位置的encoding，方便后续将文本序列补全成相同长度
        self.position_encoding = nn.Embedding(max_seq_len + 2, d_model)#+1是因为多了pad
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
    def forward(self, input_len):
        # print(input_len)
        max_len = torch.max(input_len)
        # print(max_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor( [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])

        return self.position_encoding(input_pos)
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         '''
#         x: [seq_len, batch_size, d_model]
#         '''
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        # print('attention')
        residual = context
        output = self.feed_forward(context)
        output = self.layer_norm(residual + output)  # Add & Norm
        return output, attention
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.seq_embedding = Embeddings(model_dim, vocab_size + 1, 0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
    def forward(self, inputs, inputs_len):
        # print('encoder')
        # print(inputs.size())
        output = self.seq_embedding(inputs)
        # print('seq')
        # print(inputs_len)
        output += self.pos_embedding(inputs_len)
        # print('embedding')
        self_attention_mask = padding_mask(inputs, inputs)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
        # print(output)
        return output, attentions
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout,debug=False)
        self.context_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.linear_src = nn.Linear(model_dim, model_dim)
        self.linear_temp = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=2)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
    def forward(self, dec_inputs, src_enc_output, temp_enc_output, self_attn_mask=None):
        # print('in',dec_inputs.size())
        # if self_attn_mask is not None:
        #     print(self_attn_mask.size(),dec_inputs.size())
        dec_output, self_attention = self.self_attention( dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        # judge(dec_output,'1')
        # print('out1',dec_output.size())
        residual = dec_output
        src_dec_output, src_context_attention = self.context_attention( src_enc_output, src_enc_output, dec_output)
        temp_dec_output, temp_context_attention = self.context_attention( temp_enc_output, temp_enc_output, dec_output)
        beta = self.softmax(self.linear_src(src_dec_output) + self.linear_temp(temp_dec_output))
        # print(src_dec_output.size(),temp_dec_output.size(),beta.size())
        dec_output = torch.mul(beta,src_dec_output) + torch.mul(torch.ones(beta.size()).cuda()-beta,temp_dec_output)
        # print('out2', dec_output.size())
        dec_output = self.layer_norm1(residual + dec_output)  # Add & Norm
        residual = dec_output
        dec_output = self.feed_forward(dec_output)
        dec_output = self.layer_norm1(residual + dec_output)  # Add & Norm
        return dec_output, self_attention, src_context_attention, temp_context_attention
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList( [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.seq_embedding = Embeddings(model_dim, vocab_size + 1, 0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.layer_norm = nn.LayerNorm(model_dim)
    def forward(self, inputs, inputs_len, src_enc_output, temp_enc_output):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)
        # output = self.layer_norm(output)
        # print(output.max())
        # print('-'*30)
        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)#gt：各个位置上值大于0则为1，否则为0
        # print('decoder')
        # print(self_attn_mask.size(),inputs_len)
        self_attentions = []
        src_context_attentions = []
        temp_context_attentions = []
        for decoder in self.decoder_layers:
            # print(output.size())
            output, self_attn, src_context_attn, temp_context_attn = decoder(output, src_enc_output, temp_enc_output, self_attn_mask)
            self_attentions.append(self_attn)
            src_context_attentions.append(src_context_attn)
            temp_context_attentions.append(temp_context_attn)
        return output, self_attentions, src_context_attentions, temp_context_attentions
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_max_len, temp_vocab_size, temp_max_len, tgt_vocab_size, tgt_max_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(Transformer, self).__init__()
        self.src_encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.temp_encoder = Encoder(temp_vocab_size, temp_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        #self.softmax = nn.Softmax(dim=2)
    def forward(self, src_seq, src_len, temp_seq, temp_len, tgt_seq, tgt_len):
        # print('xxx',src_seq.size(),src_len)
        src_output, src_enc_self_attn = self.src_encoder(src_seq, src_len)
        temp_output, temp_enc_self_attn = self.temp_encoder(temp_seq, temp_len)
        output, dec_self_attn, src_ctx_attn, temp_ctx_attn = self.decoder( tgt_seq, tgt_len, src_output, temp_output)
        # print(sum(output))
        output = self.linear(output)
        #output = self.softmax(output)
        return output, src_enc_self_attn, temp_enc_self_attn, dec_self_attn, src_ctx_attn, temp_ctx_attn
