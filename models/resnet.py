
import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=2, pretrained=True):
  # ImageNet으로 학습된 ResNet-50 불러옴
  weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
  model = models.resnet50(weights=weights)

  # 마지막 FC layer를 2클래스용으로 교체
  # 원래: 2048 -> 1000 (ImageNet 1000개 클래스)
  # 변경: 2048 -> 2 (물새/육지새)
  in_features = model.fc.in_features #2048
  model.fc = nn.Linear(in_features, num_classes)

  return model

def get_device():
  # 사용 가능한 디바이스 자동 선택
  if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPU
  elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon
  else:
    device = torch.device("cpu")

  print(f"사용 디바이스: {device}")
  return device


