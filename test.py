# -*- coding: utf-8 -*-
# Time    : 2023/11/2 12:56
# Author  : fanc
# File    : test.py

from monai.bundle import ConfigParser
import functools

config = ConfigParser()
config.read_config('./configs/config.yaml')

metrics_seg_list = config.get_parsed_content("VALIDATION#metrics#seg")
print(metrics_seg_list)
metrics_seg = {k.func.__name__: k for k in metrics_seg_list if type(k) == functools.partial}
print(metrics_seg)
metrics_seg.update({k.__class__.__name__: k for k in metrics_seg_list if type(k) != functools.partial})
print(metrics_seg)