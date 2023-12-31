import torch.nn as nn
from .quant_block import QuantAttentionBlock, QuantDownsample, QuantResBlock, QuantUpsample, specials, BaseQuantBlock
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
# from .fold_bn import search_fold_and_remove_bn, search_fold_and_remove_gn
from .fold_bn import search_fold_and_remove_bn

class QuantModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__()
        search_fold_and_remove_bn(model)
        # search_fold_and_remove_gn(model)
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(
        self,
        module: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(
                    module,
                    name,
                    specials[type(child_module)](
                        child_module, weight_quant_params, act_quant_params
                    ),
                )
            elif type(child_module).__name__ == "ResBlock":
                # print("### Res")
                setattr(
                    module,
                    name,
                    QuantResBlock(child_module, weight_quant_params, act_quant_params),
                )
                prev_quantmodule = getattr(module, name)
            elif type(child_module).__name__ == "Upsample":
                # print("### Upsample")
                setattr(
                    module,
                    name,
                    QuantUpsample(child_module, weight_quant_params, act_quant_params),
                )
                prev_quantmodule = getattr(module, name)
            elif type(child_module).__name__ == "Downsample":
                # print("### Upsample")
                setattr(
                    module,
                    name,
                    QuantDownsample(child_module, weight_quant_params, act_quant_params),
                )
                prev_quantmodule = getattr(module, name)
            elif type(child_module).__name__ == "AttentionBlock":
                setattr(
                    module,
                    name,
                    QuantAttentionBlock(child_module, weight_quant_params, act_quant_params),
                )
                prev_quantmodule = getattr(module, name)
            # elif isinstance(child_module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            #     # new: isinstance(child_module, (nn.Conv2d, nn.Linear, nn.Conv1d))
            #     # origin: isinstance(child_module, (nn.Conv2d, nn.Linear))
            #     setattr(
            #         module,
            #         name,
            #         QuantModule(child_module, weight_quant_params, act_quant_params),
            #     )
            #     prev_quantmodule = getattr(module, name)
            # elif isinstance(child_module, (nn.SiLU, nn.ReLU, nn.ReLU6)) or type(child_module).__name__ == "SiLU":
            #     """
            #     origin: isinstance(child_module, (nn.ReLU, nn.ReLU6))
            #     Sample batch quant_part_1w_1024uniform_disableOutAct_noSilu_1000iters.npz
            #     Inception Score:9.353286743164062
            #     FID:17.6754067118398
            #     sFID:16.800494252316184
                
            #     new: isinstance(child_module, (nn.SiLU, nn.ReLU, nn.ReLU6))
            #     Sample batch quant_part_1w_1024uniform_disableOutAct_1000iters.npz
            #     Inception Score:9.334514617919922
            #     FID:17.79372075064032
            #     sFID:16.330892220516944                
            #     """
            #     setattr(module, name,                 
            #             QuantModule(None, weight_quant_params, act_quant_params,
            #                         act_first = False, 
            #                         disable_act_quant = False, 
            #                         only_act = True, 
            #                         activation_function = child_module))

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(
                    child_module, weight_quant_params, act_quant_params
                )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, *args, **kwargs):
        if len(args)==1 and type(args[0]) in [tuple,list]:
            return self.model(*args[0])
        else:
            return self.model(*args, **kwargs)

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

    def set_cosine_embedding_layer_to_32bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(32)
        a_list[0].bitwidth_refactor(32)
        # a_list[1].bitwidth_refactor(32)
        # a_list[2].bitwidth_refactor(32)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

