import torch.nn as nn
import torch as th
from QDrop.quant.nn import avg_pool_nd, checkpoint, conv_nd
import torch.nn.functional as F
from QDrop.quant.unet import AttentionBlock, Downsample, QKVAttention, ResBlock, TimestepBlock, Upsample
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer

import math

import numpy as np

class BaseQuantBlock(nn.Module):
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
            if isinstance(m, nn.Sequential):
                self.fetch_QuantModules(m, weight_quant, act_quant)

    def fetch_QuantModules(self, sub_m, weight_quant, act_quant):
        if isinstance(sub_m, nn.Sequential):
            for sub_ms in sub_m.children():
                self.fetch_QuantModules(sub_ms, weight_quant, act_quant)
        elif isinstance(sub_m, QuantModule):
            sub_m.set_quant_state(weight_quant, act_quant)
        
            


class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self, res: ResBlock, weight_quant_params:dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm
        # QuantModule
        self.act_resnet = act_quant_params['act_resnet'] if 'act_resnet' in act_quant_params else False
        self.act_first = True # act first would cause obvious IS/FID/sFID loss
        if not self.act_resnet:
            self.in_layers = res.in_layers
            self.emb_layers = res.emb_layers
            self.out_layers = res.out_layers
            self.skip_connection = res.skip_connection
        else:
            # print(type(res.in_layers[1]))
            # print(type(res.emb_layers[0]))
            # assert isinstance(res.in_layers[1], nn.SiLU)
            # assert isinstance(res.emb_layers[0], nn.SiLU)

            # self.in_layers_silu0 = res.in_layers[1]
            # self.in_layers_silu0_activation_quantizer0 = UniformAffineQuantizer(**act_quant_params)

            
            self.in_layers = nn.Sequential(
                res.in_layers[0],
                # self.in_layers_silu0,  
                QuantModule(None, weight_quant_params, act_quant_params,
                    act_first = False, disable_act_quant = False, only_act = True, activation_function = res.in_layers[1]),
                QuantModule(res.in_layers[2], weight_quant_params, act_quant_params,
                            act_first = False, disable_act_quant = True)
            )
            # activation_function = res.emb_layers[0]

            # self.emb_layers_silu0 = res.emb_layers[0]
            
            # Emb Activation Has Huge Quantization Error!!!
            # self.emb_layers = res.emb_layers
            self.emb_layers = nn.Sequential( 
                QuantModule(None, weight_quant_params, act_quant_params,
                    act_first = False, disable_act_quant = False, only_act = True,
                    activation_function = res.emb_layers[0], split_emb = True),
                QuantModule(res.emb_layers[1], weight_quant_params, act_quant_params,
                            act_first = False, disable_act_quant = True)
            )
            # activation_function = res.emb_layers[0]
            # self.out_layers_silu0 = res.out_layers[1]
            # if self.act_resnet:
            #     self.out_layers_silu0 = nn.Sequential(
            #         self.out_layers_silu0,
            #         UniformAffineQuantizer(**act_quant_params)
            #     )
            # self.out_layers_dropout0 = res.out_layers[2]
            self.out_layers = nn.Sequential(
                res.out_layers[0],
                QuantModule(None, weight_quant_params, act_quant_params,
                    act_first = False, disable_act_quant = False, only_act = True, activation_function = res.out_layers[1]),
                res.out_layers[2],
                QuantModule(res.out_layers[3], weight_quant_params, act_quant_params,
                            act_first = False, disable_act_quant = True),
            )
            # self.out_layers[3].activation_function = nn.SiLU()

            if isinstance(res.skip_connection, nn.Identity):
                self.skip_connection = nn.Identity()
            else:
                # self.skip_connection = res.skip_connection
                self.skip_connection = QuantModule(res.skip_connection, weight_quant_params, act_quant_params, disable_act_quant = True)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )  

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)

            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class QuantUpsample(BaseQuantBlock, TimestepBlock):
    def __init__(
        self, org: Upsample, weight_quant_params:dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = org.channels
        self.use_conv = org.use_conv
        self.dims = org.dims

        self.quant_upsample_layer = True

        if org.use_conv:
            if not self.quant_upsample_layer:
                self.conv = org.conv
            else:
                self.conv = QuantModule(
                    org.conv,
                    weight_quant_params,
                    act_quant_params,
                    disable_act_quant = True
                )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class QuantDownsample(BaseQuantBlock, TimestepBlock):
    def __init__(
        self, org: Downsample, weight_quant_params:dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        super().__init__()
        self.channels = org.channels
        self.use_conv = org.use_conv
        self.dims = org.dims
        stride = 2 if org.dims != 3 else (1, 2, 2)
        
        self.quant_downsample_layer = True
        if org.use_conv:
            if self.quant_downsample_layer:
                self.op = QuantModule(org.op, weight_quant_params, act_quant_params, disable_act_quant = True)
            else:
                self.op = org.op
        else:
            self.op = org.op

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    
class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self, org: AttentionBlock, weight_quant_params:dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = org.channels
        self.num_heads = org.num_heads
        self.use_checkpoint = org.use_checkpoint

        self.norm = org.norm
        self.qkv = org.qkv
        self.act_quantizer_q = QuantModule(None, weight_quant_params, act_quant_params,
                    act_first = False, disable_act_quant = False, only_act = True)
        self.act_quantizer_k = QuantModule(None, weight_quant_params, act_quant_params,
                    act_first = False, disable_act_quant = False, only_act = True)
        self.act_quantizer_v =QuantModule(None, weight_quant_params, act_quant_params,
                    act_first = False, disable_act_quant = False, only_act = True)

        self.proj_out = QuantModule(org.proj_out, weight_quant_params, act_quant_params, disable_act_quant = True)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])

        # QKV Attention
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        q = self.act_quantizer_q(q)
        k = self.act_quantizer_q(k)
        v = self.act_quantizer_q(v)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        h = th.einsum("bts,bcs->bct", weight, v)

        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


specials = {
    ResBlock: QuantResBlock,
    Upsample: QuantUpsample,
    Downsample: QuantDownsample,
    # AttentionBlock: QuantAttentionBlock
}

