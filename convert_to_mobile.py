
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from pathlib import Path
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torch.load('./output/weights.pt')
# Set the model to evaluate mode
model.eval()

device = "cpu"
model_conv = model.to(device)


scripted_model = torch.jit.script(model_conv)
optimized_model = optimize_for_mobile(scripted_model)
spec = Path("specs/live.spec.json").read_text()
extra_files = {}
extra_files["model/live.spec.json"] = spec
optimized_model._save_for_lite_interpreter("deeplabv3-custom.ptl", _extra_files=extra_files)
print("model successfully exported - conv")