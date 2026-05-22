
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.insert(0, '/content/shortcut-learning-vision')

from data.load_waterbirds import get_waterbirds_loaders
from models.resnet import get_resnet50, get_device

# DRO 핵심: 그룹별 loss를 추적하고 worst group에 더 높은 가중치 부여
class GroupDROLoss:
  def __init__(self, n_groups=4, eta=0.01):
    self.n_groups = n_groups
    self.eta = eta  # 가중치 업데이트 step size
    self.weights = torch.ones(n_groups) / n_groups  # 초기 균등 가중치

  def compute(self, outputs, labels, metadata, device):
    self.weights = self.weights.to(device)
    group_labels = metadata[:, 0].to(device)
    groups = (labels * 2 + group_labels).long()

    criterion = nn.CrossEntropyLoss(reduction='none')
    per_sample_loss = criterion(outputs, labels)

    group_losses = torch.zeros(self.n_groups, device=device)
    for g in range(self.n_groups):
      mask = (groups == g)
      if mask.sum() > 0:
        group_losses[g] = per_sample_loss[mask].mean()

    # 가중치 업데이트: loss 높은 그룹 가중치 올리기
    self.weights = self.weights * torch.exp(self.eta * group_losses.detach())
    self.weights = self.weights / self.weights.sum()  # 정규화

    return (self.weights * group_losses).sum()


def train_one_epoch(model, loader, optimizer, dro_loss, device):
  model.train()
  total_loss = 0
  correct = 0
  total = 0

  for x, y, metadata in tqdm(loader, desc="Training"):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(x)

    loss = dro_loss.compute(outputs, y, metadata, device)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    preds = outputs.argmax(dim=1)
    correct += (preds == y).sum().item()
    total += y.size(0)

  return total_loss / len(loader), correct / total * 100


def evaluate(model, loader, device):
  model.eval()
  all_preds, all_labels, all_metadata = [], [], []

  with torch.no_grad():
    for x, y, metadata in tqdm(loader, desc="Evaluating"):
      x, y = x.to(device), y.to(device)
      outputs = model(x)
      preds = outputs.argmax(dim=1)
      all_preds.append(preds.cpu())
      all_labels.append(y.cpu())
      all_metadata.append(metadata.cpu())

  return torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_metadata)


def compute_worst_group_accuracy(preds, labels, metadata):
  group_labels = metadata[:, 0]
  groups = labels * 2 + group_labels
  group_accs = []
  for g in range(4):
    mask = (groups == g)
    if mask.sum() > 0:
      acc = (preds[mask] == labels[mask]).float().mean().item() * 100
      group_accs.append(acc)
      print(f"  그룹 {g} 정확도: {acc:.2f}%")
  worst_acc = min(group_accs)
  print(f"  Worst-group Accuracy: {worst_acc:.2f}%")
  return worst_acc


def run_group_dro(num_epochs=10, lr=1e-4, batch_size=64,
          weight_decay=1e-4, eta=0.01,
          save_path="./checkpoints/dro_best.pth"):
  os.makedirs("./checkpoints", exist_ok=True)
  device = get_device()
  train_loader, val_loader, test_loader = get_waterbirds_loaders(batch_size=batch_size)
  model = get_resnet50(num_classes=2, pretrained=True).to(device)
  dro_loss = GroupDROLoss(n_groups=4, eta=eta)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_worst_acc = 0
  for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, dro_loss, device)
    preds, labels, metadata = evaluate(model, val_loader, device)
    val_acc = (preds == labels).float().mean().item() * 100

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Acc: {val_acc:.2f}%")
    print("Val 그룹별 정확도:")
    worst_acc = compute_worst_group_accuracy(preds, labels, metadata)

    # ERM과 달리 worst-group acc 기준으로 저장
    if worst_acc > best_worst_acc:
      best_worst_acc = worst_acc
      torch.save(model.state_dict(), save_path)
      print(f"모델 저장 완료!")

  print("\n===== 최종 테스트 결과 =====")
  model.load_state_dict(torch.load(save_path))
  preds, labels, metadata = evaluate(model, test_loader, device)
  test_acc = (preds == labels).float().mean().item() * 100
  print(f"Test Accuracy: {test_acc:.2f}%")
  print("Test 그룹별 정확도:")
  compute_worst_group_accuracy(preds, labels, metadata)
  return model
