import os
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict
import look2hear.models
import yaml
from ptflops import get_model_complexity_info
from rich import print

def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)

with open("configs/tiger.yml") as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf, parser=parser)

arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
audiomodel = getattr(look2hear.models, arg_dic["audionet"]["audionet_name"])(
    sample_rate=arg_dic["datamodule"]["data_config"]["sample_rate"],
    **arg_dic["audionet"]["audionet_config"]
)
# 配置GPU为mps
device = torch.device("mps")
a = torch.randn(1, 1, 16000).to(device)
total_macs = 0
total_params = 0
model = audiomodel.to(device)
with torch.no_grad():
    macs, params = get_model_complexity_info(
        model, (16000,), as_strings=False, print_per_layer_stat=True, verbose=False
    )
print(model(a).shape)
total_macs += macs
total_params += params
print("MACs: ", total_macs / 10.0 ** 9)
print("Params: ", total_params / 10.0 ** 6)