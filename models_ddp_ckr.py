import pdb
import copy
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from tslearn import metrics
import commons
import modules
import attentions
import monotonic_align
from modules import LinearNorm, ConvBlock, SwishBlock

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
import pdb


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class LearnableUpsampling(nn.Module):
    def __init__(
        self,
        d_predictor=192,
        kernel_size=3,
        dropout=0.0,
        conv_output_size=8,
        dim_w=4,
        dim_c=2,
        max_seq_len=1000,
    ):
        super(LearnableUpsampling, self).__init__()
        self.max_seq_len = max_seq_len

        self.conv_w = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_w = SwishBlock(conv_output_size + 2, dim_w, dim_w)
        self.linear_w = LinearNorm(dim_w * d_predictor, d_predictor, bias=True) 
        self.softmax_w = nn.Softmax(dim=2)

        self.conv_c = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_c = SwishBlock(conv_output_size + 2, dim_c, dim_c)

        self.linear_einsum = LinearNorm(dim_c * dim_w, d_predictor)
        self.layer_norm = nn.LayerNorm(d_predictor)

        self.proj_o = LinearNorm(194, 192 * 2)

    def forward(self, duration, V, src_len, src_mask, max_src_len):

        batch_size = duration.shape[0]
        mel_len = torch.round(duration.sum(-1)).type(torch.LongTensor).to(V.device)
        mel_len = torch.clamp(mel_len, max=self.max_seq_len)
        max_mel_len = mel_len.max().item()
        mel_mask = self.get_mask_from_lengths(mel_len, max_mel_len)

        src_mask_ = src_mask.unsqueeze(1).expand(
            -1, mel_mask.shape[1], -1
        )
        mel_mask_ = mel_mask.unsqueeze(-1).expand(
            -1, -1, src_mask.shape[1]
        ) 
        attn_mask = torch.zeros(
            (src_mask.shape[0], mel_mask.shape[1], src_mask.shape[1])
        ).to(V.device)

        attn_mask = attn_mask.masked_fill(src_mask_, 1.0)

        attn_mask = attn_mask.masked_fill(mel_mask_, 1.0)

        attn_mask = attn_mask.bool()

        e_k = torch.cumsum(duration, dim=1)
        s_k = e_k - duration

        e_k = e_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        s_k = s_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        t_arange = (
            torch.arange(1, max_mel_len + 1, device=V.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, max_src_len)
        )

        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(
            attn_mask, 0
        )
        W = self.swish_w(S, E, self.conv_w(V))
        W = W.masked_fill(src_mask_.unsqueeze(-1), -np.inf)
        W = self.softmax_w(W)
        W = W.masked_fill(mel_mask_.unsqueeze(-1), 0.0)
        W = W.permute(0, 3, 1, 2)

        C = self.swish_c(S, E, self.conv_c(V))

        upsampled_rep = self.linear_w(
            torch.einsum("bqtk,bkh->bqth", W, V).permute(0, 2, 1, 3).flatten(2)
        ) + self.linear_einsum(
            torch.einsum("bqtk,btkp->bqtp", W, C).permute(0, 2, 1, 3).flatten(2)
        )

        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(mel_mask.unsqueeze(-1), 0)
        upsampled_rep = self.proj_o(upsampled_rep)
        return upsampled_rep, mel_mask, mel_len, W

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device))
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      n_languages=1):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.in_dim = hidden_channels + (n_languages > 1) * n_languages

    self.encoder = attentions.Encoder(
      self.in_dim,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
  def forward(self, x, x_lengths, l=None):
    x = self.emb(x) * math.sqrt(self.hidden_channels)
    if l != None:
      x = torch.cat((x, l.unsqueeze(1).expand(x.size(0),x.size(1),-1)), dim=-1)
    x = torch.transpose(x, 1, -1)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    return x, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    encoded_speaker=False,
    n_speakers=0,
    n_languages=1,
    gin_channels=0,
    use_sdp=False,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.encoded_speaker = encoded_speaker
    self.n_speakers = n_speakers
    self.n_languages = n_languages
    self.gin_channels = gin_channels
    self.distance_map = np.loadtxt('ckpts/distance_matrix_final_normalize.txt')

    self.use_sdp = use_sdp

    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_languages=n_languages)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    self.dp = DurationPredictor(hidden_channels+(n_languages > 1) * n_languages, 256, 3, 0.5, gin_channels=gin_channels)

    self.learnable_upsampling = LearnableUpsampling(d_predictor=192+2, kernel_size=3, dropout=0.0, conv_output_size=8, dim_w=4, dim_c=2, max_seq_len=3000)

    if (n_speakers > 1) and (self.encoded_speaker is None):
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
    
    if n_languages > 1:
      self.emb_l = nn.Embedding(n_languages, n_languages)

  def forward(self, x, x_lengths, y, y_lengths, d, spk=None, lidx=None,pitch=None, pitch_lengths=None):
    if self.encoded_speaker is not None:
      g = spk.unsqueeze(-1)
    elif self.n_speakers > 1:
      g = self.emb_g(spk).unsqueeze(-1)
    else:
      g = None

    if lidx is not None:
      l = self.emb_l(lidx)
    else:
      l = None

    x_emb, x_mask = self.enc_p(x, x_lengths, l=l)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)

    logw = self.dp(x_emb, x_mask, g=g)
    w = torch.exp(logw) * x_mask

    w_ = d.unsqueeze(1)
    logw_ = torch.log(w_ + 1e-6) * x_mask

    l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
        x_mask
    ) 

    upsampled_rep, p_mask, _, W = self.learnable_upsampling(
        d,
        x_emb.transpose(1, 2),
        x_lengths,
        ~(x_mask.squeeze(1).bool()),
        x_lengths.max(),
    )

    p_mask = ~p_mask
    m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)

    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    return o, l_length, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, x, x_lengths, spk=None, lidx=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    if self.encoded_speaker is not None:
      g = spk.unsqueeze(-1)
    elif self.n_speakers > 1:
      g = self.emb_g(spk).unsqueeze(-1)
    else:
      g = None

    if lidx is not None:
      l = self.emb_l(lidx)
    else:
      l = None
    x, x_mask = self.enc_p(x, x_lengths, l=l)

    logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    upsampled_rep, p_mask, _, W = self.learnable_upsampling(
        w.squeeze(1),
        x.transpose(1, 2),
        x_lengths,
        ~(x_mask.squeeze(1).bool()),
        x_mask.shape[-1],
    )
    p_mask = ~p_mask
    m_p, logs_p = torch.split(upsampled_rep.transpose(1,2), 192, dim=1)
    y_mask = p_mask.unsqueeze(1)

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, y_mask, (z, z_p, m_p, logs_p)


  def infer_origin(self, x, x_lengths, spk=None, lidx=None, duration=None, noise_scale=1, length_scale=1., noise_scale_w=1., max_len=None):
    if self.encoded_speaker is not None:
      g = spk.unsqueeze(-1)
    elif self.n_speakers > 1:
      g = self.emb_g(spk).unsqueeze(-1)
    else:
      g = None

    if lidx is not None:
      l = self.emb_l(lidx)
    else:
      l = None
    x, x_mask = self.enc_p(x, x_lengths, l=l)

    logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale

    current_frames = torch.round(w).sum().item()
    current_sum = w.sum().item()
    current_length_scale = length_scale 
    new_length_scale = current_length_scale * (duration / current_sum)
    return new_length_scale, current_frames


  def infer_text(self, frames,x, x_lengths, spk, lidx=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    if self.encoded_speaker is not None:
      g = spk.unsqueeze(-1)
    elif self.n_speakers > 1:
      g = self.emb_g(spk).unsqueeze(-1)
    else:
      g = None

    if lidx is not None:
      l = self.emb_l(lidx)
    else:
      l = None
    x, x_mask = self.enc_p(x, x_lengths, l=l)
    logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale

    w_sum = w.sum().item()
    if w_sum-frames > 26:
        judge = False
        lengths = True
    elif w_sum-frames < -26 :
        judge = False
        lengths=False
    else :
        judge = True
        lengths = None  

    return judge, lengths, w_sum


  def generate_kr_list(self,kr_x, kr_frame, kr_vowel):
    kr_list = []
    kr_frame_list = kr_frame.tolist()
    kr_x_list = kr_x.tolist()
    i = 0
    while i < len(kr_x_list):
        value = kr_x_list[i]
        frame_length = kr_frame_list[i]
        
        if value in kr_vowel:
            kr_list.extend([value] * int(frame_length))
            i += 1
        elif value == 0:
            kr_list.extend([1] * int(kr_frame_list[i]))
            i += 1
        else:
            kr_list.extend([1] * int(frame_length))
            i += 1
    return kr_list
  def infer_ensh(self, x, x_lengths, spk=None, lidx=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    en_vowel = [43, 47, 51, 63, 109, 110, 111, 116, 123, 126, 127, 142, 175, 178]
    special_vowel = 198
    if self.encoded_speaker is not None:
      g = spk.unsqueeze(-1)
    elif self.n_speakers > 1:
      g = self.emb_g(spk).unsqueeze(-1)
    else:
      g = None

    if lidx is not None:
      l = self.emb_l(lidx)
    else:
      l = None

    kr_x = x.squeeze()
    x, x_mask = self.enc_p(x, x_lengths, l=l)

    logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    kr_frame = torch.round(w).squeeze()

    en_list = self.generate_eg_list(kr_x,kr_frame,en_vowel, special_vowel)
    return en_list

  def generate_eg_list(self, eg_x, eg_frame, eg_vowel, special_vowel):
    eg_list = []
    eg_frame_list = eg_frame.tolist()
    eg_x_list = eg_x.tolist()
    i = 0 
    while i < len(eg_x_list):
        value = eg_x_list[i]
        frame_length = eg_frame_list[i]
        
        if value in eg_vowel:
            eg_list.extend([value] * int(frame_length))
            i += 1
            if i < len(eg_x_list)-1 and int(eg_x_list[i+1]) == special_vowel :
              eg_list.extend([value] * int(eg_frame_list[i]))
              i += 1
              eg_list.extend([value] * int(eg_frame_list[i]))
              i += 1
        elif value == 0:
            eg_list.extend([1] * int(eg_frame_list[i]))
            i += 1
        else:
            eg_list.extend([1] * int(frame_length))
            i += 1
    return eg_list
  def custom_distance(self,x,y):
      kr_vowel_indices = {v: i for i, v in enumerate([88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108])}
      en_vowel_indices = {v: i for i, v in enumerate([43, 47, 51, 63, 109, 110, 111, 116, 123, 126, 127, 142, 175, 178])}
      x_val = x.item() if isinstance(x, np.ndarray) else x
      y_val = y.item() if isinstance(y, np.ndarray) else y
      x_idx = en_vowel_indices.get(x_val, None)
      y_idx = kr_vowel_indices.get(y_val, None)
      if x_idx is not None and y_idx is not None:
          return self.distance_map[x_idx, y_idx]
      elif x_idx is None and y_idx is None:
          return 5
      else :
          return 25

  def caculate_score(self,rep, kr_list,eg_list):
    path, alignment = metrics.dtw_path_from_metric(eg_list, kr_list, metric=self.custom_distance,global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
    return path, alignment

  def infer_score(self, x, x_lengths, rep,kr_list, spk, lidx=None, noise_scale=1, length_scale=1, noise_scale_w=1.,origin_frame=1):
    kr_vowel = [88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]

    if self.encoded_speaker is not None:
      g = spk.unsqueeze(-1)
    elif self.n_speakers > 1:
      g = self.emb_g(spk).unsqueeze(-1)
    else:
      g = None

    if lidx is not None:
      l = self.emb_l(lidx)
    else:
      l = None

    eg_x = x.squeeze()
    x, x_mask = self.enc_p(x, x_lengths, l=l)

    logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    eg_frame = torch.round(w).squeeze()
    eg_frame_sum =w.sum().item()
    if (origin_frame <250) and abs(origin_frame - eg_frame_sum) >= 30:
      return 0,1
    if abs(origin_frame - eg_frame_sum) >= 40:
      return 0,1
    final_speed = (origin_frame / (eg_frame_sum)) * length_scale
    final_speed
    w = torch.exp(logw) * x_mask * final_speed
    eg_frame = torch.round(w).squeeze()
    eg_frame_sum =w.sum().item()
    eg_list = self.generate_kr_list(eg_x, eg_frame, kr_vowel)

    path, align_dtw = self.caculate_score(rep, kr_list, eg_list)
    return (rep, align_dtw, eg_frame_sum,final_speed), path


  def find_closest(self, target, value, start_idx):
    backward = None
    forward = None

    for i in range(start_idx, -1, -1):
        if target[i] == value:
            backward = i
            break

    for i in range(start_idx, len(target)):
        if target[i] == value:
            forward = i
            break

    if backward is None and forward is not None:
        return forward
    elif backward is not None and forward is None:
        return backward
    elif backward is not None and forward is not None:
        if (start_idx - backward) <= (forward - start_idx):
            return backward
        else:
            return forward
    else:
        return None

  def infer_sl(self, x, x_lengths, silence, spk, lidx=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None): 
      if self.encoded_speaker is not None:
        g = spk.unsqueeze(-1)
      elif self.n_speakers > 1:
        g = self.emb_g(spk).unsqueeze(-1)
      else:
        g = None

      if lidx is not None:
        l = self.emb_l(lidx)
      else:
        l = None
      eg_x = x.squeeze()
      x, x_mask = self.enc_p(x, x_lengths, l=l)

      logw = self.dp(x, x_mask, g=g)

      w = torch.exp(logw) * x_mask * length_scale
      silence_intervals = silence
      adjustments = []
      for start, end in silence_intervals:
          adjusted_length = end - start
          print(silence_intervals, adjusted_length)
          if adjusted_length == 0: 
            continue
          judge=True
          for idx in range(len(eg_x)):
              if eg_x[idx] == 3 and judge==True:
              if eg_x[idx] == 3 and ((w[:,:,:idx+1].sum().item()+25) > start > (w[:,:,:idx+1].sum().item()-25)):
                  w[0,0,idx] = adjusted_length
                  adjustments.append((idx, adjusted_length))
                  judge = False
                  break
          if judge :
            for idl in range(len(eg_x)):
              if (w[:,:,:idl+1].sum().item()) > start:
                closest_idx = self.find_closest(eg_x, 16, idl)
                new_elements = torch.tensor([4,0,3, 0])
                part1 = eg_x[:closest_idx+2]
                part2 = new_elements
                part3 = eg_x[closest_idx+2:]
                eg_x = torch.cat([part1.cuda(), part2.cuda(), part3.cuda()], dim=0).cuda()
                eg_x = eg_x.unsqueeze(0)
                x, x_mask = self.enc_p(eg_x, x_lengths, l=l)
                eg_x = eg_x.squeeze()
                logw = self.dp(x, x_mask, g=g)
                w = torch.exp(logw) * x_mask * length_scale
                w[0,0,closest_idx + 4] = adjusted_length
                for idxs, adjusted_lengths in adjustments:
                  w[0, 0, idxs] = adjusted_lengths
                adjustments.append((closest_idx + 4, adjusted_length))
                break
      upsampled_rep, p_mask, _, W = self.learnable_upsampling(
          w.squeeze(1),
          x.transpose(1, 2),
          x_lengths,
          ~(x_mask.squeeze(1).bool()),
          x_mask.shape[-1],
      )
      p_mask = ~p_mask
      m_p, logs_p = torch.split(upsampled_rep.transpose(1,2), 192, dim=1)
      y_mask = p_mask.unsqueeze(1)

      z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
      z = self.flow(z_p, y_mask, g=g, reverse=True)
      o = self.dec((z * y_mask)[:,:,:max_len], g=g)
      return o, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

