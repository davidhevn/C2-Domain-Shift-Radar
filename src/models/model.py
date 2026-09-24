import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, Any, Optional

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CIFARResNet18(nn.Module):
    """
    Standard ResNet-18 architecture modified for 32x32 CIFAR images (3x3 initial conv, no maxpool).
    """
    def __init__(self, num_classes=10):
        super(CIFARResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        logits = self.linear(features)
        return logits, features

class ModelWrapper(nn.Module):
    """
    Encapsulates normalizer, model backbone, feature extraction and evaluation.
    """
    def __init__(self, backbone: nn.Module, device: Optional[torch.device] = None):
        super(ModelWrapper, self).__init__()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = backbone.to(self.device)
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(self.device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.normalize(x)
        return self.backbone(norm_x)

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        x = x.to(self.device)
        logits, features = self.forward(x)
        probs = F.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)
        return {
            "logits": logits,
            "features": features,
            "probs": probs,
            "preds": preds,
            "confidences": confidences
        }

def load_model(weights_path: Optional[str] = None, device: Optional[torch.device] = None) -> ModelWrapper:
    """
    Instantiates CIFARResNet18 and optionally loads weights.
    If no checkpoint exists, loads standard torchvision weights adapted for CIFAR.
    """
    device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = CIFARResNet18(num_classes=10)
    
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        # Pretrained backbone fallback using torchvision
        try:
            tv_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Adapt first conv and linear
            tv_model.fc = nn.Linear(tv_model.fc.in_features, 10)
            
            # Wrapper class for standard torchvision resnet
            class TVResNetWrapper(nn.Module):
                def __init__(self, tv_net):
                    super().__init__()
                    self.tv_net = tv_net
                def forward(self, x):
                    x = self.tv_net.conv1(x)
                    x = self.tv_net.bn1(x)
                    x = self.tv_net.relu(x)
                    x = self.tv_net.maxpool(x)
                    x = self.tv_net.layer1(x)
                    x = self.tv_net.layer2(x)
                    x = self.tv_net.layer3(x)
                    x = self.tv_net.layer4(x)
                    x = self.tv_net.avgpool(x)
                    features = torch.flatten(x, 1)
                    logits = self.tv_net.fc(features)
                    return logits, features

            return ModelWrapper(TVResNetWrapper(tv_model), device=device)
        except Exception:
            pass

    return ModelWrapper(model, device=device)
