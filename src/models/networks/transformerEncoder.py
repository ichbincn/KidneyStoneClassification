#把ResNetEncoder.py中的代码和Transfor.py中的代码结合
#ResNet的输出加上positionEncoding.py中的结果作为Transformer的输入
from typing import Sequence, Tuple, Union
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from nnunet.network_architecture.neural_network import SegmentationNetwork
