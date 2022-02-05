import torch
from torch import nn
import torchvision.models as models
import io

model = models.resnet50(pretrained = True)
model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

weights = torch.load("/content/model_best.pth",map_location='cpu')
model.load_state_dict(weights)
model.eval()
m = torch.jit.script(model)
torch.jit.save(m, 'scriptmodule2.pt') 
