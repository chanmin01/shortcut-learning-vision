
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset

def get_transforms(split):
  if split == "train":
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
  else:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def get_waterbirds_loaders(root_dir="./data/waterbirds", batch_size=64):
  # Waterbirds 데이터셋 자동 다운로드
  dataset = get_dataset(dataset="waterbirds", download=True, root_dir=root_dir)

  # train / val / test 세 묶음으로 나누기
  train_data = dataset.get_subset("train", transform=get_transforms("train"))
  val_data = dataset.get_subset("val", transform=get_transforms("val"))
  test_data = dataset.get_subset("test", transform=get_transforms("test"))

  # 64장씩 묶어서 AI한테 전달하는 DataLoader 생성
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  print(f"Train: {len(train_data)}개 | Val: {len(val_data)}개 | Test: {len(test_data)}개")
  return train_loader, val_loader, test_loader

