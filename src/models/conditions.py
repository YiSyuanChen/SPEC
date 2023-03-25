""" Conditional Modules """
import collections
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
import pdb

##### Config #####


class ConditionalAdapterConfig(NamedTuple):
    input_size: int
    hidden_size: int
    condition_size: int
    act: str
    init_range: float
    num_beams: int


class ConditionalLNConfig(NamedTuple):
    input_size: int
    condition_size: int
    num_beams: int


##### Conditional Modules #####


class ConditionalLN(nn.Module):
    def __init__(self, ln: nn.LayerNorm, config: ConditionalLNConfig):
        super().__init__()
        self.ln = ln
        self.gamma = nn.Parameter(
            torch.zeros(config.condition_size, config.input_size))
        self.beta = nn.Parameter(
            torch.zeros(config.condition_size, config.input_size))
        nn.init.normal_(self.gamma)
        nn.init.zeros_(self.beta)

        self.num_beams = config.num_beams
        self.conditions = None

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        layer_normed = self.ln(hidden_states)

        # Conditional layer norm
        gamma = torch.mm(self.conditions, self.gamma).unsqueeze(1)
        beta = torch.mm(self.conditions, self.beta).unsqueeze(1)
        if self.conditions.shape[0] != batch_size:
            # NOTE: for beam search decoding : [d1b1, d1b2, d2,b1, ...]
            gamma = torch.repeat_interleave(gamma, self.num_beams, dim=0)
            beta = torch.repeat_interleave(beta, self.num_beams, dim=0)
        condition_layer_normed = torch.mul(layer_normed, gamma) + beta

        return condition_layer_normed


class ConditionalAdapter(nn.Module):
    def __init__(self, config: ConditionalAdapterConfig):
        super().__init__()
        # Conditional down project
        self.down_project = nn.Parameter(
            torch.zeros(config.input_size, config.hidden_size))
        self.down_gamma = nn.Parameter(
            torch.zeros(config.condition_size, config.hidden_size))
        self.down_beta = nn.Parameter(
            torch.zeros(config.condition_size, config.hidden_size))
        nn.init.normal_(self.down_project, std=config.init_range)
        nn.init.normal_(self.down_gamma)
        nn.init.zeros_(self.down_beta)

        # Activation
        self.activation = ACT2FN[config.act]

        # Conditional up project
        self.up_project = nn.Parameter(
            torch.zeros(config.hidden_size, config.input_size))
        self.up_gamma = nn.Parameter(
            torch.zeros(config.condition_size, config.input_size))
        self.up_beta = nn.Parameter(
            torch.zeros(config.condition_size, config.input_size))
        nn.init.normal_(self.up_project, std=config.init_range)
        nn.init.normal_(self.up_gamma)
        nn.init.zeros_(self.up_beta)

        # Layer Norm
        ln_config = ConditionalLNConfig(input_size=config.input_size,
                                        condition_size=config.condition_size,
                                        num_beams=config.num_beams)
        self.ln = ConditionalLN(ln=nn.LayerNorm(config.input_size),
                                config=ln_config)

        self.num_beams = config.num_beams
        self.conditions = None

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]

        # Conditional down project
        down_project = torch.cat(batch_size * [self.down_project.unsqueeze(0)])
        down_gamma = torch.mm(self.conditions, self.down_gamma).unsqueeze(1)
        down_beta = torch.mm(self.conditions, self.down_beta).unsqueeze(1)
        if self.conditions.shape[0] != batch_size:
            # NOTE: for beam search decoding : [d1b1, d1b2, d2b1, ...]
            down_gamma = torch.repeat_interleave(down_gamma,
                                                 self.num_beams,
                                                 dim=0)
            down_beta = torch.repeat_interleave(down_beta,
                                                self.num_beams,
                                                dim=0)
        condition_down_project = torch.mul(down_project,
                                           down_gamma) + down_beta
        down_projected = torch.matmul(hidden_states, condition_down_project)

        # Activation
        activated = self.activation(down_projected)

        # Conditional up project
        up_project = torch.cat(batch_size * [self.up_project.unsqueeze(0)])
        up_gamma = torch.mm(self.conditions, self.up_gamma).unsqueeze(1)
        up_beta = torch.mm(self.conditions, self.up_beta).unsqueeze(1)
        if self.conditions.shape[0] != batch_size:
            # NOTE: for beam search decoding : [d1b1, d1b2, d2b1, ...]
            up_gamma = torch.repeat_interleave(up_gamma, self.num_beams, dim=0)
            up_beta = torch.repeat_interleave(up_beta, self.num_beams, dim=0)
        condition_up_project = torch.mul(up_project, up_gamma) + up_beta
        up_projected = torch.matmul(activated, condition_up_project)

        # Conditional layer norm
        self.ln(up_projected)

        return up_projected + hidden_states


class ConditionalFF(nn.Module):
    def __init__(self, ff: nn.Linear, config: ConditionalAdapterConfig):
        super().__init__()
        self.ff = ff
        self.adapter = ConditionalAdapter(config).to(ff.weight.device)

    def forward(self, hidden_states):
        return self.adapter(self.ff(hidden_states))


class ConditionalSwitchFF(nn.Module):
    def __init__(self, ff: nn.Linear, config: ConditionalAdapterConfig):
        super().__init__()
        self.ff = ff
        self.adapter_dict = nn.ModuleDict({
            "main":
            ConditionalAdapter(config).to(ff.weight.device),
            "side":
            ConditionalAdapter(config).to(ff.weight.device),
        })
        self.adapter_flag = "main"

    def forward(self, hidden_states):
        return self.adapter_dict[self.adapter_flag](self.ff(hidden_states))

class ConditionalSplitFF(nn.Module):
    def __init__(self, ff: nn.Linear, config: ConditionalAdapterConfig):
        super().__init__()
        self.ff = ff
        self.adapter_dict = nn.ModuleDict({
            "main":
            ConditionalAdapter(config).to(ff.weight.device),
            "side":
            ConditionalAdapter(config).to(ff.weight.device),
        })
        self.adapter_flag = "main"
        self.weighting_mean = nn.Parameter(torch.tensor(0.5).to(ff.weight.device)) 

    def forward(self, hidden_states):
        if self.adapter_flag == "main":
            hidden_states_main = self.adapter_dict['main'](self.ff(hidden_states))
            with torch.no_grad():
                hidden_states_side = self.adapter_dict['side'](self.ff(hidden_states))
        elif self.adapter_flag == "side":
            with torch.no_grad():
                hidden_states_main = self.adapter_dict['main'](self.ff(hidden_states))
            hidden_states_side = self.adapter_dict['side'](self.ff(hidden_states))
        else:
            raise ValueError("Invalid adapter flag.")

        noise = torch.normal(mean=0, std=0.001, size=(1,1)).to(self.ff.weight.device)
        weighting = self.weighting_mean + noise
        hidden_states = hidden_states_main*weighting + hidden_states_side*(1-weighting)

        return hidden_states


##### Insert Functions #####


def _conditional_ff(config: ConditionalAdapterConfig):
    return lambda ff: ConditionalFF(ff, config=config)

def _conditional_switch_ff(config: ConditionalAdapterConfig):
    return lambda ff: ConditionalSwitchFF(ff, config=config)

def _conditional_split_ff(config: ConditionalAdapterConfig):
    return lambda ff: ConditionalSplitFF(ff, config=config)


def insert_conditional_adapters(model_args, data_args, model):

    # Set config
    adapter_config = ConditionalAdapterConfig(
        input_size=model.config.d_model,
        hidden_size=model_args.adapter_size,
        condition_size=model_args.adapter_condition_size,
        act=model_args.adapter_act,
        init_range=model_args.adapter_init_range,
        num_beams=data_args.num_beams
        if data_args.num_beams else model.config.num_beams)

    # Set positions
    adapter_positions = []
    if model_args.adapter_positions == "full_dec":
        for layer in model.model.decoder.layers:
            adapter_positions.append(layer)
    elif model_args.adapter_positions == "full":
        for layer in model.model.encoder.layers:
            adapter_positions.append(layer)
        for layer in model.model.decoder.layers:
            adapter_positions.append(layer)
    else:
        raise ValueError("Invalid adapter positions.")

    # Set replace function
    if model_args.adapter_type == "default":
        replace_func = _conditional_ff
    elif model_args.adapter_type == "sw":
        replace_func = _conditional_switch_ff
    elif model_args.adapter_type == "sp":
        replace_func = _conditional_split_ff
    else:
        raise ValueError("Invalid adapter type.")

    # Insert adapters
    ModuleName = collections.namedtuple("ModuleName", "parent, child")
    for layer in adapter_positions:
        # Record modules to be replaced
        replace_module_names = []
        for name, sub_module in layer.named_modules():
            if isinstance(sub_module, nn.Linear):
                name_split = name.split(".")
                if name_split[-1] in ['fc2', 'out_proj']:
                    replace_module_names.append(
                        ModuleName(".".join(name_split[:-1]), name_split[-1]))

        # Replace modules accord to names
        for parent_name, child_name in replace_module_names:
            for name, sub_module in layer.named_modules():
                if name == parent_name:
                    setattr(
                        sub_module, child_name,
                        replace_func(adapter_config)(getattr(
                            sub_module, child_name)))
                    break

    return model
