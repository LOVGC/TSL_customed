import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1) # xf: [B, T', C], T' = T//2 +1 因为只保留了正频率部分
   
    # find period by amplitudes: here we assume that the periodic features are basically constant
    # in different batch and channel, so we mean out these two dimensions, getting a list frequency_list with shape[T] 
    # each element at pos t of frequency_list denotes the overall amplitude at frequency (t)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0 # 这里还把 0 频率给抹去了

    _, top_list = torch.topk(frequency_list, k) # 这里 _ 是 values, top_list 是 indices，这里指的就是离散的周期。可以理解成每隔多少个点，重复一次
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list # 这里 x.shape[1] 就是 T, 只的是总时间长度，那么 period 就是对应的就是这个 T 里面有多少个周期。有多少个周期，就把这个时序数据分割成多少段。
                                    # 总结，这里的 period 指的就是每个 segment 的长度是多少, 是根据频率计算出来的。
    
    #Here,the 2nd item returned has a shape of [B, top_k],representing the biggest top_k amplitudes 
    # for each piece of data, with N features being averaged.
    return period, abs(xf).mean(-1)[:, top_list] # 2nd item shape: [B, top_k]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len  #sequence length 
        self.pred_len = configs.pred_len #prediction length
        self.k = configs.top_k # k denotes how many top frequencies are 
                               # taken into consideration

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )  # 这里，可以发现，输入，输出的 dimension 又一样了。

    def forward(self, x):
        B, T, N = x.size() # #B: batch size  T: length of time series  N:number of features。这里 N 指的应该是每一个 time point 对应的那个 vector 的长度
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            # 感觉核心概念还是 (B,T,N) 把这个 T 维度拆成 (p, f) 的意思。
            # 我理解的就是 fix 一个 B 和一个 N，我们得到的是一个 1D 的时间序列，
            # 然后把这个 1D 的序列变成 2D 的 (p, f)。这个就是把 (B,T,N) 这个 T 维度拆成 (p, f) 的意思
            # 你完全可以写个 codesnippet 来验证一下 reshape 的过程。而且你还可以用 einops 来 implement
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()  # reshape 和 permute 后 out: (B, N, fi, pi)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N) # out: (B, T, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :]) # 这里的 T 只保留 self.seq_len + self.pred_len 这么长。
        res = torch.stack(res, dim=-1) # 这里 res 的 shape 应该是 (B, T, N, k), k=num_kernels
        # adaptive aggregation  这里代码实现看起来还挺复杂的。需要研究一下。
        period_weight = F.softmax(period_weight, dim=1)  # period_weight: [B, k], k is the top k amplitudes 
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)  # period_weight: [B, 1, 1, k] -> [B, T, N, k] k=num_kernels
        res = torch.sum(res * period_weight, -1) # res: [B, T, N]
        # residual connection
        res = res + x # res: [B, T, N]
        return res #  res: [B, T, N]
 

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # params init
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        #stack TimesBlock for e_layers times to form the main part of TimesNet, named model
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)]) # 堆叠了 e_layers 个 TimesBlock
        
        #embedding & normalization
        # enc_in is the encoder input size, the number of features for a piece of data
        # d_model is the dimension of embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers # number of TimesBlock layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        #define some layers for different tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer(这个技巧来自论文： Non-stationary Transformer)
        # 其实就是对时间序列做了一个归一化处理，这里好像是 instance norm.
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, C]
        x_enc = x_enc.sub(means) # (B, T, C)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev) # (B, T, C)

        # embedding： projecting a number to a N-channel vector
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,N]， N is d_model，x_mark_enc 可以是 None
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension [B,pred_len+seq_len,N], 这里必须对 nn.Linear 有深刻的理解：nn.Linear 是对最后一个维度做变换的。
        
        # TimesNet: pass through TimesBlock for self.layer times each with layer normalization
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back  #[B,T,d_model]-->[B,T,c_out]
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))) # 乘回标准差
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))) # 加回均值
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out)) # (B, seq_len, N = d_model)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)  # 这里 act() 就是 gelu, (B, seq_length, N=d_model)
        output = self.dropout(output) # (B, seq_length, N=d_model)

        # 这个是干嘛的？ 需要研究一下从这里往下的细节。

        # zero-out padding embeddings:The primary role of x_mark_enc in the code is to 
        # zero out the embeddings for padding positions in the output tensor through 
        # element-wise multiplication, helping the model to focus on meaningful data 
        # while disregarding padding. 这里 zero out 啥意思？
        output = output * x_mark_enc.unsqueeze(-1) 
        
        output = output.reshape(output.shape[0], -1)  
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
