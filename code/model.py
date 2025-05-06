import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import torchvision.transforms as transforms

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm 
import copy

# import torch_npu
# from torch_npu.contrib import transfer_to_npu

ModelResPrediction = namedtuple('ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t
  
def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
    ):
        super().__init__()

        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x, time):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x


class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise"
    ):
        super().__init__()
        self.condition = condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
    
        self.unet0 = Unet(dim,
                        init_dim=init_dim,
                        out_dim=out_dim,
                        dim_mults=dim_mults,
                        channels=channels,
                        resnet_block_groups=resnet_block_groups,
                        learned_variance=learned_variance,
                        learned_sinusoidal_cond=learned_sinusoidal_cond,
                        random_fourier_features=random_fourier_features,
                        learned_sinusoidal_dim=learned_sinusoidal_dim,
                        condition=condition)

    def forward(self, x, time):
        if self.objective == "pred_noise":
            time = time[1]
        elif self.objective == "pred_res":
            time = time[0]
        return [self.unet0(x, time)]

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "lamda":
        x = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        alphas = 1 - y
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)

    return alphas*sum_scale

def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        delta_end = 2.0e-3,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta= 0,
        condition=False,
        sum_scale=None,
        test_res_or_noise="None",
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.test_res_or_noise = test_res_or_noise
        self.delta_end = delta_end

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            betas = (torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2)
        elif beta_schedule == "squaredcos_cap_v2":
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
            
        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        delta_cumsum_prev = F.pad(delta_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = 0
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = 0
        betas_cumsum = torch.sqrt(betas2_cumsum) 

        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('delta', delta)
        register_buffer('delta_cumsum', delta_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', 
            (betas2_cumsum_prev)/(betas2_cumsum)*(alphas - delta) + (betas2)/(betas2_cumsum)*(alphas_cumsum_prev - delta_cumsum_prev)
        )
        register_buffer('posterior_mean_coef3', delta + betas2/betas2_cumsum*(1 - delta_cumsum_prev))
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6


    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t - (1-extract(self.delta_cumsum,t,x_t.shape)) * x_input - (extract(self.alphas_cumsum, t, x_t.shape)-1) * pred_res) /extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise + extract(self.delta_cumsum, t, x_t.shape) * x_input )/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise, x_input):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise + extract(self.delta_cumsum, t, x_t.shape) * x_input
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t, x_input):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res + extract(self.delta, t, x_t.shape) * x_input -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t): 
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, task=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,[t,t])

        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise, x_input)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = self.predict_start_from_res_noise(x, t, pred_res, pred_noise, x_input)
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t):
        preds = self.model_predictions(x_input, x, t)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_input, x=x, t=batched_times)
        noise = torch.randn_like(x) if t > 0 else 0. 
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(x_input, img, t)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def first_order_sample(self, x_input, shape, last=True, task=None): 
        x_input = x_input[0] 
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
   
        if self.condition:
            img = self.betas_cumsum[-1] * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"
        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_input, img, time_cond, task)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next] 
            betas2_div_betas_cumsum = betas_cumsum-betas_cumsum_next  
 
            if type == "use_pred_noise": 
                img = img - alpha*pred_res + delta*x_input - betas2_div_betas_cumsum * pred_noise  
                
            elif type == "use_x_start":
                img = q*img + \
                    (1-q)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*q)*pred_res + \
                    (delta_cumsum*q-delta_cumsum_next)*x_input + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise
                    
            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)


    @torch.no_grad()
    def second_order_sample(self, x_input, shape, last=True, task=None): 
        x_input = x_input[0] 
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        r = 0.5
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            time_internal = int((1-r) * time + r * time_next)
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            time_cond_internal = torch.full((batch,), time_internal, device=device, dtype=torch.long)
            time_cond_next = torch.full((batch,), time_next, device=device, dtype=torch.long)

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_internal = self.alphas_cumsum[time_internal]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha_u = alpha_cumsum-alpha_cumsum_internal
            alpha_t = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_internal = self.delta_cumsum[time_internal]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta_u = delta_cumsum-delta_cumsum_internal
            delta_t = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_internal = self.betas2_cumsum[time_internal]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2_u = betas2_cumsum-betas2_cumsum_internal
            betas2_t = betas2_cumsum-betas2_cumsum_next
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_internal = self.betas_cumsum[time_internal]
            betas_cumsum_next = self.betas_cumsum[time_next]
            betas_u = betas_cumsum-betas_cumsum_internal
            betas_t = betas_cumsum-betas_cumsum_next

            preds_time_cond = self.model_predictions(x_input, img, time_cond)
            pred_res_time_cond = preds_time_cond.pred_res
            pred_noise_time_cond = preds_time_cond.pred_noise
            x_start_time_cond = preds_time_cond.pred_x_start

            if time_next < 0:
                img = x_start_time_cond
                if not last:
                    img_list.append(img)
                continue

            img_u = img + delta_u*x_input - alpha_u*pred_res_time_cond - betas_u * pred_noise_time_cond

            preds_time_cond_internal = self.model_predictions(x_input, img_u.clone().detach(), time_cond_internal)
            pred_res_time_cond_internal = preds_time_cond_internal.pred_res
            pred_noise_time_cond_internal = preds_time_cond_internal.pred_noise
            x_start_time_cond_internal = preds_time_cond_internal.pred_x_start

            img_target = img + delta_t*x_input - alpha_t*pred_res_time_cond - betas_u * pred_noise_time_cond 
            - 1/(2*r)*alpha_t * (pred_res_time_cond_internal - pred_res_time_cond)
            - 1/(2*r)*betas_t * (pred_noise_time_cond_internal - pred_noise_time_cond)

            img = img_target.clone().detach()

            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)
 
    @torch.no_grad()
    def third_order_sample(self, x_input, shape, last=True, task=None): 
        x_input = x_input[0] 
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        r1 = 1/3
        r2 = 2/3
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            time_u = int((1-r1) * time + r1 * time_next)
            time_s = int((1-r2) * time + r2 * time_next)
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            time_cond_u = torch.full((batch,), time_u, device=device, dtype=torch.long)
            time_cond_s = torch.full((batch,), time_s, device=device, dtype=torch.long)
            time_cond_next = torch.full((batch,), time_next, device=device, dtype=torch.long)

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_u = self.alphas_cumsum[time_u]
            alpha_cumsum_s = self.alphas_cumsum[time_s]            
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha_u = alpha_cumsum-alpha_cumsum_u
            alpha_s = alpha_cumsum-alpha_cumsum_s
            alpha_t = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_u = self.delta_cumsum[time_u]
            delta_cumsum_s = self.delta_cumsum[time_s]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta_u = delta_cumsum-delta_cumsum_u
            delta_s = delta_cumsum-delta_cumsum_s
            delta_t = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_u = self.betas2_cumsum[time_u]
            betas2_cumsum_s = self.betas2_cumsum[time_s]            
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2_u = betas2_cumsum-betas2_cumsum_u
            betas2_s = betas2_cumsum-betas2_cumsum_s
            betas2_t = betas2_cumsum-betas2_cumsum_next
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_u = self.betas_cumsum[time_u]
            betas_cumsum_s = self.betas_cumsum[time_s]
            betas_cumsum_next = self.betas_cumsum[time_next]
            betas_u = betas_cumsum-betas_cumsum_u
            betas_s = betas_cumsum-betas_cumsum_s            
            betas_t = betas_cumsum-betas_cumsum_next

            preds_time_cond = self.model_predictions(x_input, img, time_cond)
            pred_res_time_cond = preds_time_cond.pred_res
            pred_noise_time_cond = preds_time_cond.pred_noise
            x_start_time_cond = preds_time_cond.pred_x_start

            if time_next < 0:
                img = x_start_time_cond
                if not last:
                    img_list.append(img)
                continue

            img_u = img + delta_u*x_input - alpha_u*pred_res_time_cond - betas_u * pred_noise_time_cond
            preds_time_u = self.model_predictions(x_input, img_u, time_cond_u)
            pred_res_time_u = preds_time_u.pred_res
            pred_noise_time_u = preds_time_u.pred_noise
            x_start_time_u = preds_time_u.pred_x_start

            img_s = img + delta_s*x_input - alpha_s*pred_res_time_cond - betas_s * pred_noise_time_cond
            preds_time_s = self.model_predictions(x_input, img_s, time_cond_s)
            pred_res_time_s = preds_time_s.pred_res
            pred_noise_time_s = preds_time_s.pred_noise
            x_start_time_s = preds_time_s.pred_x_start

            
            D1_res = pred_res_time_u - pred_res_time_cond
            D1_eps = pred_noise_time_u - pred_noise_time_cond
            D2_res = (2/(r1*r2*(r2-r1)))*(r1*pred_res_time_s - r2*pred_res_time_u + (r2-r1)*pred_res_time_cond)
            D2_eps = (2/(r1*r2*(r2-r1)))*(r1*pred_noise_time_s - r2*pred_noise_time_u + (r2-r1)*pred_noise_time_cond)

            img_target = img + delta_t*x_input - alpha_t*pred_res_time_cond - betas_u * pred_noise_time_cond 
            - 1/(2*r1)*alpha_t * D1_res - 1/(2*r1)*betas_t * D1_eps
            - 1/6*alpha_t*D2_res - 1/6*betas_t*D2_eps

            img = img_target 

            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)
  

    def grad_and_value(self, x_prev, x_0_hat, y0, x_res):  
        x_pre = x_prev
        difference = y0- (x_0_hat + x_res)
        norm = torch.linalg.norm(difference)  
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, norm

    def first_order_UPS(self, x_input, shape, last=True, task=None): 
        x_input = x_input[0] 
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
   
        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input +  self.betas_cumsum[-1] * torch.randn(shape, device=device) 
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            img = img.requires_grad_()

            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_input, img, time_cond, task)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]

            betas2_div_betas_cumsum = betas2 / betas_cumsum 

            norm_grad, norm = self.grad_and_value(x_prev=img, x_0_hat=x_start, y0=x_input, x_res = pred_res)
            pred_noise = pred_noise + betas_cumsum / norm * norm_grad
         
            if type == "use_pred_noise": 
                img = img - alpha*pred_res + delta*x_input - betas2_div_betas_cumsum * pred_noise  
                
            elif type == "use_x_start":
                img = q*img + \
                    (1-q)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*q)*pred_res + \
                    (delta_cumsum*q-delta_cumsum_next)*x_input + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise
                    
            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_lists
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    def second_order_UPS(self, x_input, shape, last=True, task=None): 
        x_input = x_input[0] 
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        r = 0.5
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            img = img.requires_grad_()

            time_internal = int((1-r) * time + r * time_next)
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            time_cond_internal = torch.full((batch,), time_internal, device=device, dtype=torch.long)
            time_cond_next = torch.full((batch,), time_next, device=device, dtype=torch.long)

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_internal = self.alphas_cumsum[time_internal]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha_u = alpha_cumsum-alpha_cumsum_internal
            alpha_t = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_internal = self.delta_cumsum[time_internal]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta_u = delta_cumsum-delta_cumsum_internal
            delta_t = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_internal = self.betas2_cumsum[time_internal]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2_u = betas2_cumsum-betas2_cumsum_internal
            betas2_t = betas2_cumsum-betas2_cumsum_next
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_internal = self.betas_cumsum[time_internal]
            betas_cumsum_next = self.betas_cumsum[time_next]
            betas_u = betas_cumsum-betas_cumsum_internal
            betas_t = betas_cumsum-betas_cumsum_next

            preds_time_cond = self.model_predictions(x_input, img, time_cond)
            pred_res_time_cond = preds_time_cond.pred_res
            pred_noise_time_cond = preds_time_cond.pred_noise
            x_start_time_cond = preds_time_cond.pred_x_start

            if time_next < 0:
                img = x_start_time_cond
                if not last:
                    img_list.append(img)
                continue
 
            norm_grad, norm = self.grad_and_value(x_prev=img, x_0_hat=x_start_time_cond, y0=x_input, x_res = pred_res_time_cond)
            pred_noise_time_cond = pred_noise_time_cond + betas_cumsum / norm * norm_grad

            img_u = img + delta_u*x_input - alpha_u*pred_res_time_cond - betas_u * pred_noise_time_cond

            img_u = img_u.clone().detach_()
            img_u = img_u.requires_grad_()

            preds_time_cond_internal = self.model_predictions(x_input, img_u, time_cond_internal)
            pred_res_time_cond_internal = preds_time_cond_internal.pred_res
            pred_noise_time_cond_internal = preds_time_cond_internal.pred_noise
            x_start_time_cond_internal = preds_time_cond_internal.pred_x_start

            norm_grad_internal, norm_internal = self.grad_and_value(x_prev=img_u, x_0_hat=x_start_time_cond_internal, y0=x_input, x_res = pred_res_time_cond_internal)
            pred_noise_time_cond_internal = pred_noise_time_cond_internal + betas_cumsum_internal / norm_internal * norm_grad_internal

            img_target = img + delta_t*x_input - alpha_t*pred_res_time_cond - betas_t * pred_noise_time_cond 
            - 1/(2*r)*alpha_t * (pred_res_time_cond_internal - pred_res_time_cond)
            - 1/(2*r)*betas_t * (pred_noise_time_cond_internal - pred_noise_time_cond)

            img = img_target.clone().detach_()
 
            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    def third_order_UPS(self, x_input, shape, last=True, task=None): 
        x_input = x_input[0] 
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        r1 = 1/3
        r2 = 2/3
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            img = img.requires_grad_()

            time_u = int((1-r1) * time + r1 * time_next)
            time_s = int((1-r2) * time + r2 * time_next)
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            time_cond_u = torch.full((batch,), time_u, device=device, dtype=torch.long)
            time_cond_s = torch.full((batch,), time_s, device=device, dtype=torch.long)
            time_cond_next = torch.full((batch,), time_next, device=device, dtype=torch.long)

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_u = self.alphas_cumsum[time_u]
            alpha_cumsum_s = self.alphas_cumsum[time_s]            
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha_u = alpha_cumsum-alpha_cumsum_u
            alpha_s = alpha_cumsum-alpha_cumsum_s
            alpha_t = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_u = self.delta_cumsum[time_u]
            delta_cumsum_s = self.delta_cumsum[time_s]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta_u = delta_cumsum-delta_cumsum_u
            delta_s = delta_cumsum-delta_cumsum_s
            delta_t = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_u = self.betas2_cumsum[time_u]
            betas2_cumsum_s = self.betas2_cumsum[time_s]            
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2_u = betas2_cumsum-betas2_cumsum_u
            betas2_s = betas2_cumsum-betas2_cumsum_s
            betas2_t = betas2_cumsum-betas2_cumsum_next
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_u = self.betas_cumsum[time_u]
            betas_cumsum_s = self.betas_cumsum[time_s]
            betas_cumsum_next = self.betas_cumsum[time_next]
            betas_u = betas_cumsum-betas_cumsum_u 
            betas_s = betas_cumsum-betas_cumsum_s 
            betas_t = betas_cumsum-betas_cumsum_next 

            preds_time_cond = self.model_predictions(x_input, img, time_cond)
            pred_res_time_cond = preds_time_cond.pred_res
            pred_noise_time_cond = preds_time_cond.pred_noise
            x_start_time_cond = preds_time_cond.pred_x_start

            if time_next < 0:
                img = x_start_time_cond
                if not last:
                    img_list.append(img)
                continue
 
            norm_grad, norm = self.grad_and_value(x_prev=img, x_0_hat=x_start_time_cond, y0=x_input, x_res = pred_res_time_cond)
            pred_noise_time_cond = pred_noise_time_cond + betas_cumsum / norm * norm_grad
            img = img.detach_()

            img_u = img + delta_u*x_input - alpha_u*pred_res_time_cond - betas_u * pred_noise_time_cond
            img_u = img_u.clone().detach_()
            img_u = img_u.requires_grad_()
            preds_time_cond_u = self.model_predictions(x_input, img_u, time_cond_u)
            pred_res_time_cond_u = preds_time_cond_u.pred_res
            pred_noise_time_cond_u = preds_time_cond_u.pred_noise
            x_start_time_cond_u = preds_time_cond_u.pred_x_start

            norm_grad_u, norm_u = self.grad_and_value(x_prev=img_u, x_0_hat=x_start_time_cond_u, y0=x_input, x_res = pred_res_time_cond_u)
            pred_noise_time_cond_u = pred_noise_time_cond_u + betas_cumsum_u / norm_u * norm_grad_u
            img_u = img_u.detach_()

            img_s = img + delta_s*x_input - alpha_s*pred_res_time_cond - betas_s * pred_noise_time_cond 
            - r2/(2*r1)*alpha_s * (pred_res_time_cond_u - pred_res_time_cond)
            - r2/(2*r1)*betas_s * (pred_noise_time_cond_u - pred_noise_time_cond)

            img_s = img_s.clone().detach_()
            img_s = img_s.requires_grad_()
            preds_time_cond_s = self.model_predictions(x_input, img_s, time_cond_s)
            pred_res_time_cond_s = preds_time_cond_s.pred_res
            pred_noise_time_cond_s = preds_time_cond_s.pred_noise
            x_start_time_cond_s = preds_time_cond_s.pred_x_start

            norm_grad_s, norm_s = self.grad_and_value(x_prev=img_s, x_0_hat=x_start_time_cond_s, y0=x_input, x_res = pred_res_time_cond_s)
            pred_noise_time_cond_s = pred_noise_time_cond_s +  betas_cumsum_s / norm_s * norm_grad_s
            img_s = img_s.detach_()

            D2_res = (2/(r1*r2*(r2-r1)))*(r1*pred_res_time_cond_s - r2*pred_res_time_cond_u + (r2-r1)*pred_res_time_cond)
            D2_eps = (2/(r1*r2*(r2-r1)))*(r1*pred_noise_time_cond_s - r2*pred_noise_time_cond_u + (r2-r1)*pred_noise_time_cond)


            img = img + delta_t*x_input - alpha_t*pred_res_time_cond - betas_t * pred_noise_time_cond 
            - 1/(2*r1)*alpha_t * (pred_res_time_cond_u - pred_res_time_cond) - 1/(2*r1)*betas_t * (pred_noise_time_cond_u - pred_noise_time_cond)
            - 1/6*alpha_t*D2_res - 1/6*betas_t*D2_eps

            img = img.clone().detach_() 
 
            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)


    def sample(self, x_input=None, batch_size=16, last=True, task=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.second_order_UPS
        # self.first_order_sample           1st
        # self.first_order_UPS              UPS_1st
        # self.second_order_sample          2nd
        # self.second_order_UPS             UPS_2nd
        # self.third_order_sample           3rd
        # self.third_order_UPS              UPS_3rd
        if self.condition:
            x_input = 2 * x_input - 1
            x_input = x_input.unsqueeze(0)

            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)

        gen_samples = sample_fn(x_input, size, last=last, task=task)[1]
        gen_samples = gen_samples.detach()
        return gen_samples

    def q_sample(self, x_start, x_res, condition, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise -
            extract(self.delta_cumsum, t, x_start.shape) * condition
        )

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  
            x_input = 2 * imgs[1] - 1 
            x_start = 2 * imgs[0] - 1  
            task = None

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start
        b, c, h, w = x_start.shape
        x = self.q_sample(x_start, x_res, x_input, t, noise=noise)
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in,[t,t])

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)
            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)
            pred_res = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t,x_input)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t,x_input)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            for i in range(len(model_out)):
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
            return loss_list

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, *args, **kwargs)


if __name__ == '__main__':
    print('Hello World')