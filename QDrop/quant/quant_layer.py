import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        channel_wise: bool = False,
        scale_method: str = "minmax",
        leaf_param: bool = False,
        prob: float = 1.0,
    ):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        # if self.sym:
        #     raise NotImplementedError
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        """if leaf_param, use EMA to set scale"""
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        """mse params"""
        self.scale_method = "mse"
        self.one_side_dist = None
        self.num = 100

        """for activation quantization"""
        self.running_min = None
        self.running_max = None

        """do like dropout"""
        self.prob = prob
        self.is_training = False

        # quant_params for each timestep
        self.timewise_quant_mode = False
        self.fast_mode = False
        self.now_t = None
        self.timestep_params = {}

    def change_timestep(self, t):
        assert self.timewise_quant_mode
        assert type(t) == int
        if self.now_t is not None:
            self.timestep_params[self.now_t] = [
                self.delta,
                self.zero_point,
                self.inited,
            ]
        if t not in self.timestep_params:
            self.timestep_params[t] = [1, 0, False]
        self.delta, self.zero_point, self.inited = self.timestep_params[t]
        self.now_t = t

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            # print(f"{self} init start")
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(
                    x.clone().detach(), self.channel_wise
                )
            else:
                self.delta, self.zero_point = self.init_quantization_scale(
                    x.clone().detach(), self.channel_wise
                )
        if self.fast_mode:
            x_sim = (
                (x / self.delta)
                .round_()
                .clamp_(-self.zero_point, self.n_levels - 1 - self.zero_point)
                .mul_(self.delta)
            )
            return x_sim
        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            if x.dim() > 1:
                y = torch.flatten(x, 1)
                return y.mean(1)
            else:
                y = x
                return y.mean(0)

    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    # @profile
    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise and x.dim() > 1:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        # if zero_point.ndim == 0:
        #     print(delta, zero_point)
        return x_float_q

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2**self.n_bits - 1)
            # enumerate zp
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    # @profile
    def perform_1D_search(self, x):
        if self.channel_wise:
            y = None
            if x.dim() > 1:
                y = torch.flatten(x, 1)
                x_min, x_max = torch._aminmax(y, 1)
            else:
                y = x
                x_min, x_max = torch._aminmax(y, 0)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        # for scale in torch.linspace(0.2, 1.2, self.num):
        #     thres = xrange * scale
        if not self.channel_wise:
            # accelerated calibration
            thres = (
                xrange / self.num * torch.arange(1, self.num + 1, device=x.device)
            )  # shape n
            new_min = torch.zeros_like(thres) if self.one_side_dist == "pos" else -thres
            new_max = torch.zeros_like(thres) if self.one_side_dist == "neg" else thres
            quant_min, quant_max = 0, self.n_levels - 1
            scale = (new_max - new_min) / float(quant_max - quant_min)  # shape n
            scale = torch.max(scale, self.eps)
            zero_point = -torch.round(new_min / scale)
            zero_point = torch.clamp_(zero_point, quant_min, quant_max).view(-1, 1)
            scale = scale.view(-1, 1)
            # print(scale, zero_point)
            scores = []
            batch = 8
            for i in range(0, self.num, batch):
                x_int = (
                    x.view(1, -1) / scale[i : i + batch]
                ).round_()  # shape batch, size
                x_int.clamp_(
                    -zero_point[i : i + batch],
                    self.n_levels - 1 - zero_point[i : i + batch],
                )
                x_sim = x_int.mul_(scale[i : i + batch])
                score = (
                    x_sim.sub_(x.view(1, -1)).abs_().pow_(2.4).mean(1)
                )  # shape batch
                scores.append(score)
                del x_sim
            # print(scores)
            ind = torch.argmin(torch.hstack(scores))
            best_min1 = new_min[ind]
            best_max1 = new_max[ind]
            return best_min1, best_max1

        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == "pos" else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == "neg" else thres

            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            # if not self.channel_wise:
            #     print(i, score)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method != "mse":
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
            )
        if (
            self.one_side_dist != "no" or self.sym
        ):  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        with torch.no_grad():
            x_min, x_max = self.get_x_min_x_max(x)
            scale, zero_point = self.calculate_qparams(x_min, x_max)
        return scale, zero_point

    def init_quantization_scale(
        self, x_clone: torch.Tensor, channel_wise: bool = False
    ):
        if channel_wise and x_clone.dim() > 1:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 32, "bitwidth not supported"
        self.n_bits = refactored_bit
        self.n_levels = 2**self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return "bit={}, is_training={}, inited={}".format(
            self.n_bits, self.is_training, self.inited
        )


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(
        self,
        org_module: Union[nn.Conv2d, nn.Conv1d, nn.Linear, nn.GroupNorm],
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        act_first=False,
        disable_act_quant=True,
        only_act=False,
        activation_function=StraightThrough(),
        split_emb=False,
    ):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        elif isinstance(org_module, nn.GroupNorm):
            self.fwd_kwargs = dict(
                num_groups = org_module.num_groups
            )
            self.fwd_func = F.group_norm
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(
                stride=org_module.stride, 
                padding=org_module.padding,
                dilation=org_module.dilation, 
                groups=org_module.groups
            )
            self.fwd_func = F.conv1d  
        self.only_act = only_act

        if not only_act:     
            self.weight = org_module.weight
            self.org_weight = org_module.weight.data.clone()
            if org_module.bias is not None:
                self.bias = org_module.bias
                self.org_bias = org_module.bias.data.clone()
            else:
                self.bias = None
                self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = activation_function
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.act_first = act_first
        self.split_embed = split_emb
        if self.split_embed:
            self.shift_quantizer = UniformAffineQuantizer(**act_quant_params)
            self.scale_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.name_tag = str(org_module) if org_module is not None else "Act Module"

    def forward(self, input: torch.Tensor):
        is_analysis = False
        out = input
        if self.only_act:
            out = torch.tensor(self.activation_function(out))
            tmp_out = out.clone()
            if self.split_embed and self.use_act_quant and not self.disable_act_quant:
                scale_out, shift_out = torch.chunk(out, 2, dim = 1)
                scale_out = self.scale_quantizer(scale_out)
                shift_out = self.shift_quantizer(shift_out)
                out = torch.cat([scale_out, shift_out], dim=1)
            if self.use_act_quant and not self.disable_act_quant and not self.split_embed:
                out = self.act_quantizer(out)
            if is_analysis:
                col1 = str(F.mse_loss(tmp_out, out).item())
                col2 = str(tmp_out.max().item())
                col3 = str(tmp_out.min().item())
                col4 = str(out.max().item())
                col5 = str(out.min().item())
                val = col1 + "\t" + col2 + "\t" + col3 + "\t" + col4 + "\t" + col5
                with open("outputs/analysis/Layer_analysis_data_file.txt",'a') as f:
                    f.write(val + "\n")
    
                with open("outputs/analysis/Layer_analysis_file.txt",'a') as f:
                    f.write("[Layer analysis in " + self.name_tag + "]\n[ACT-MSE is " + str(F.mse_loss(tmp_out, out).item()) + "]\n")

            return out
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        if self.act_first:
            out = self.activation_function(out)
            if self.use_act_quant and not self.disable_act_quant:
                if is_analysis:
                    col1 = str(F.mse_loss(out, self.act_quantizer(out)).item())
                    col2 = str(out.max().item())
                    col3 = str(out.min().item())
                    col4 = str(self.act_quantizer(out).max().item())
                    col5 = str(self.act_quantizer(out).min().item())
                    val = col1 + "\t" + col2 + "\t" + col3 + "\t" + col4 + "\t" + col5
                    with open("outputs/analysis/Layer_analysis_data_file.txt",'a') as f:
                        f.write(val + "\n")
        
                    with open("outputs/analysis/Layer_analysis_file.txt",'a') as f:
                        f.write("[Layer analysis in " + self.name_tag + "]\n[ACT-MSE is " + str(F.mse_loss(out, self.act_quantizer(out)).item()) + "]\n")
                out = self.act_quantizer(out)
        out = self.fwd_func(out, weight=weight, bias=bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.

        
        # if F.mse_loss(out, self.act_quantizer(out)).item() > 0.01:
        #     return out
        # else:
        #     return self.act_quantizer(out)
        if self.use_act_quant and not self.act_first and not self.disable_act_quant and is_analysis:
            col1 = str(F.mse_loss(out, self.act_quantizer(out)).item())
            col2 = str(out.max().item())
            col3 = str(out.min().item())
            col4 = str(self.act_quantizer(out).max().item())
            col5 = str(self.act_quantizer(out).min().item())
            val = col1 + "\t" + col2 + "\t" + col3 + "\t" + col4 + "\t" + col5
            with open("outputs/analysis/Layer_analysis_data_file.txt",'a') as f:
                f.write(val + "\n")
  
            with open("outputs/analysis/Layer_analysis_file.txt",'a') as f:
                f.write("[Layer analysis in " + self.name_tag + "]\n[ACT-MSE is " + str(F.mse_loss(out, self.act_quantizer(out)).item()) + "]\n")
        # if self.disable_act_quant and not self.act_first:
        #     return out
        if not self.act_first:
            out = self.activation_function(out)
            if self.use_act_quant and not self.disable_act_quant:
                out = self.act_quantizer(out)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return "wbit={}, abit={}, disable_act_quant={}".format(
            self.weight_quantizer.n_bits,
            self.act_quantizer.n_bits,
            self.disable_act_quant,
        )
    
# class QuantModule(nn.Module):
#     """
#     Quantized Module that can perform quantized convolution or normal convolution.
#     To activate quantization, please use set_quant_state function.
#     """
#     def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
#                  act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff'):
#         super(QuantModule, self).__init__()
#         self.weight_quant_params = weight_quant_params
#         self.act_quant_params = act_quant_params
#         if isinstance(org_module, nn.Conv2d):
#             self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
#                                    dilation=org_module.dilation, groups=org_module.groups)
#             self.fwd_func = F.conv2d
#         elif isinstance(org_module, nn.Conv1d):
#             self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
#                                    dilation=org_module.dilation, groups=org_module.groups)
#             self.fwd_func = F.conv1d
#         else:
#             self.fwd_kwargs = dict()
#             self.fwd_func = F.linear
#         self.weight = org_module.weight
#         self.org_weight = org_module.weight.data.clone()
#         if org_module.bias is not None:
#             self.bias = org_module.bias
#             self.org_bias = org_module.bias.data.clone()
#         else:
#             self.bias = None
#             self.org_bias = None
#         # de-activate the quantized forward default
#         self.use_weight_quant = False
#         self.use_act_quant = False
#         self.act_quant_mode = act_quant_mode
#         self.disable_act_quant = disable_act_quant
#         # initialize quantizer
#         self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params)
#         if self.act_quant_mode == 'qdiff':
#             self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)
#         self.split = 0

#         self.activation_function = StraightThrough()
#         self.ignore_reconstruction = False
        
#         self.extra_repr = org_module.extra_repr

#         self.name_tag = str(org_module)

#     def forward(self, input: torch.Tensor, split: int = 0):
#         if split != 0 and self.split != 0:
#             assert(split == self.split)
#         elif split != 0:
#             print(f"split at {split}!")
#             self.split = split
#             self.set_split()

#         if not self.disable_act_quant and self.use_act_quant:
#             raw_input = input.clone()
#             if self.split != 0:
#                 if self.act_quant_mode == 'qdiff':
#                     input_0 = self.act_quantizer(input[:, :self.split, :, :])
#                     input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
#                 input = torch.cat([input_0, input_1], dim=1)
#             else:
#                 if self.act_quant_mode == 'qdiff':
#                     input = self.act_quantizer(input)
#             col1 = str(F.mse_loss(raw_input, input).item())
#             col2 = str(raw_input.max().item())
#             col3 = str(raw_input.min().item())
#             col4 = str(input.max().item())
#             col5 = str(input.min().item())
#             val = col1 + "\t" + col2 + "\t" + col3 + "\t" + col4 + "\t" + col5
#             with open("outputs/analysis/Layer_analysis_data_file.txt",'a') as f:
#                 f.write(val + "\n")
  
#             with open("outputs/analysis/Layer_analysis_file.txt",'a') as f:
#                 f.write("[Layer analysis in " + self.name_tag + "]\n[ACT-MSE is " + str(F.mse_loss(raw_input, input).item()) + "]\n")
        
#         if self.use_weight_quant:
#             if self.split != 0:
#                 weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
#                 weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
#                 weight = torch.cat([weight_0, weight_1], dim=1)
#             else:
#                 weight = self.weight_quantizer(self.weight)
#             bias = self.bias
#         else:
#             weight = self.org_weight
#             bias = self.org_bias
#         out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
#         out = self.activation_function(out)

#         return out

#     def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
#         self.use_weight_quant = weight_quant
#         self.use_act_quant = act_quant

#     def set_split(self):
#         self.weight_quantizer_0 = UniformAffineQuantizer(**self.weight_quant_params)
#         if self.act_quant_mode == 'qdiff':
#             self.act_quantizer_0 = UniformAffineQuantizer(**self.act_quant_params)

#     def set_running_stat(self, running_stat: bool):
#         if self.act_quant_mode == 'qdiff':
#             self.act_quantizer.running_stat = running_stat
#             if self.split != 0:
#                 self.act_quantizer_0.running_stat = running_stat
