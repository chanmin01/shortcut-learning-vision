
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.insert(0, '/content/shortcut-learning-vision')

from data.load_waterbirds import get_waterbirds_loaders
from models.resnet import get_resnet50, get_device

def train_one_epoch(model, loader, optimizer, criterion, device):
  model.train()
  total_loss = 0
  correct = 0
  total = 0

  for x, y, metadata in tqdm(loader, desc="Training"):

    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(x)

    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    preds = outputs.argmax(dim=1)
    correct += (preds == y).sum().item()
    total += y.size(0)
  return total_loss / len(loader), correct / total * 100

def evaluate(model, loader, criterion, device):

  model.eval()
  total_loss = 0
  all_preds, all_labels, all_metadata = [], [], []
  
  with torch.no_grad():
    for x, y, metadata in tqdm(loader, desc="Evaluating"):

      x, y = x.to(device), y.to(device)
      outputs = model(x)
      loss = criterion(outputs, y)
      total_loss += loss.item()
      preds = outputs.argmax(dim=1)
      all_preds.append(preds.cpu())
      all_labels.append(y.cpu())
      all_metadata.append(metadata.cpu())
  all_preds = torch.cat(all_preds)
  all_labels = torch.cat(all_labels)
  all_metadata = torch.cat(all_metadata)
  return total_loss / len(loader), all_preds, all_labels, all_metadata

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

def run_erm(num_epochs=10, lr=1e-4, batch_size=64,
            weight_decay=1e-4,
            save_path="./checkpoints/erm_best.pth"):
  os.makedirs("./checkpoints", exist_ok=True)
  device = get_device()
  train_loader, val_loader, test_loader = get_waterbirds_loaders(batch_size=batch_size)
  model = get_resnet50(num_classes=2, pretrained=True).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  best_val_acc = 0
  for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, preds, labels, metadata = evaluate(model, val_loader, criterion, device)
    val_acc = (preds == labels).float().mean().item() * 100
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), save_path)
      print(f"모델 저장 완료!")
  print("\n===== 최종 테스트 결과 =====")
  model.load_state_dict(torch.load(save_path))
  _, preds, labels, metadata = evaluate(model, test_loader, criterion, device)
  test_acc = (preds == labels).float().mean().item() * 100
  print(f"Test Accuracy: {test_acc:.2f}%")
  compute_worst_group_accuracy(preds, labels, metadata)
  return model
