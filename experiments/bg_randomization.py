"""
experiments/bg_randomization.py
================================
Background Randomization — 팀 고유 Contribution
Investigating Shortcut Learning in Deep Vision Models

핵심 아이디어:
  ERM은 학습 데이터에서 "물새 = 물 배경"이라는 shortcut을 외워버림.
  해결책: 학습 중 배경 영역을 랜덤 노이즈 or 단색으로 덮어서
  모델이 배경에 의존하지 못하게 강제함.

  배경 마스크를 어떻게 구하냐?
  → WILDS Waterbirds 데이터셋은 자체적으로 배경/새 분리 정보가 없음.
  → 대신 이미지 중앙 영역을 새로 간주하고
    주변부(상하좌우 margin)를 배경으로 취급하는 근사 방식 사용.
  → 또는 전체 이미지에 랜덤 색조(hue) shift를 주는 방식도 병행.

사용법:
  from experiments.bg_randomization import run_bg_randomization
  model, results = run_bg_randomization()
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from models.resnet import get_resnet50, get_device

# ── 찬민 코드 import ─────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_waterbirds import get_waterbirds_loaders

# ── 재현성 시드 고정 ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  하이퍼파라미터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG = {
    "num_epochs":    50,
    "batch_size":    64,
    "learning_rate": 1e-4,
    "weight_decay":  1e-4,
    "num_classes":   2,

    # Background Randomization 파라미터
    # center_ratio: 이미지에서 새(foreground)로 간주하는 중앙 영역 비율
    #   0.5 → 이미지 중앙 50% 영역은 새, 나머지 주변부는 배경
    "center_ratio":  0.5,

    # aug_prob: 배경 randomization을 적용할 확률
    #   0.8 → 80% 확률로 배경을 랜덤화, 20%는 원본 유지
    "aug_prob":      0.8,

    # bg_mode: 배경을 어떻게 바꿀지
    #   "noise"  → 가우시안 랜덤 노이즈로 배경 교체
    #   "color"  → 랜덤 단색으로 배경 교체
    #   "both"   → noise / color 중 랜덤하게 선택
    "bg_mode":       "both",

    # Waterbirds 그룹 정보
    "num_groups":    4,
    "group_names": [
        "Group 0: 물새 + 물 배경  (다수)",
        "Group 1: 물새 + 땅 배경  (소수) ★",
        "Group 2: 육지새 + 물 배경 (소수) ★",
        "Group 3: 육지새 + 땅 배경 (다수)",
    ],

    "checkpoint_dir":  "./checkpoints",
    "results_dir":     "./results",
    "checkpoint_name": "bg_randomization_best.pth",
}

# ImageNet 정규화 통계
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Background Randomization 핵심 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def randomize_background(
    images:       torch.Tensor,   # (B, 3, H, W) — 정규화된 텐서
    center_ratio: float,
    aug_prob:     float,
    bg_mode:      str,
    device:       torch.device,
) -> torch.Tensor:
    """
    배치 내 이미지들의 배경 영역을 랜덤하게 교체.

    동작 원리:
      1. 이미지를 정규화 역변환 (0~1 범위로 복원)
      2. 중앙 center_ratio 영역 = 새(foreground) → 보존
      3. 나머지 주변부 = 배경(background) → 랜덤 교체
         - "noise" : 가우시안 노이즈
         - "color" : 랜덤 단색 (R, G, B 각각 랜덤)
         - "both"  : 위 두 가지 중 랜덤 선택
      4. 다시 정규화 적용

    Args:
        images:       배치 이미지 텐서 (정규화된 상태)
        center_ratio: 중앙 foreground 영역 비율 (0~1)
        aug_prob:     augmentation 적용 확률
        bg_mode:      배경 교체 방식 ("noise" / "color" / "both")
        device:       연산 디바이스

    Returns:
        augmented images, 같은 shape (B, 3, H, W)
    """
    B, C, H, W = images.shape

    # ── 정규화 역변환 (denormalize) ──────────────────────────────────────
    mean = MEAN.view(1, 3, 1, 1).to(device)
    std  = STD.view(1, 3, 1, 1).to(device)
    imgs = images.clone() * std + mean   # 0~1 범위로 복원
    imgs = imgs.clamp(0, 1)

    # ── 중앙 foreground 마스크 생성 ───────────────────────────────────────
    # 중앙 50% 영역: h_start~h_end, w_start~w_end
    margin_h = int(H * (1 - center_ratio) / 2)
    margin_w = int(W * (1 - center_ratio) / 2)
    h_start, h_end = margin_h, H - margin_h
    w_start, w_end = margin_w, W - margin_w

    # foreground_mask: 중앙=True, 배경=False  shape: (1, 1, H, W)
    fg_mask = torch.zeros(1, 1, H, W, device=device, dtype=torch.bool)
    fg_mask[:, :, h_start:h_end, w_start:w_end] = True

    # ── 각 이미지에 aug_prob 확률로 배경 randomization 적용 ───────────────
    result = imgs.clone()
    for i in range(B):
        if random.random() > aug_prob:
            continue   # 이 이미지는 원본 유지

        # 배경 교체 방식 선택
        mode = bg_mode
        if mode == "both":
            mode = random.choice(["noise", "color"])

        if mode == "noise":
            # 가우시안 랜덤 노이즈 (0~1 범위로 clip)
            bg = torch.randn(C, H, W, device=device) * 0.3 + 0.5
            bg = bg.clamp(0, 1)
        else:  # "color"
            # 랜덤 단색: R, G, B 각각 0~1 랜덤값
            r = random.random()
            g = random.random()
            b = random.random()
            bg = torch.zeros(C, H, W, device=device)
            bg[0] = r
            bg[1] = g
            bg[2] = b

        # 배경 영역만 교체 (foreground는 원본 유지)
        fg = fg_mask.squeeze(0).expand(C, H, W)   # (3, H, W)
        result[i] = torch.where(fg, imgs[i], bg)

    # ── 다시 정규화 ───────────────────────────────────────────────────────
    result = (result - mean) / std

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  모델 로딩 (찬민의 models/resnet.py 사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_model(num_classes: int, device: torch.device) -> nn.Module:
    """찬민의 get_resnet50 사용 — 마지막 레이어 2클래스로 교체된 ResNet-50"""
    model = get_resnet50(num_classes=num_classes, pretrained=True)
    return model.to(device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Epoch 단위 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    criterion:    nn.Module,
    optimizer:    optim.Optimizer,
    device:       torch.device,
    epoch:        int,
    config:       dict,
) -> dict:
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    group_correct = np.zeros(config["num_groups"])
    group_total   = np.zeros(config["num_groups"])

    for batch_idx, batch in enumerate(loader):
        # WILDS DataLoader: (x, y, metadata) 형태로 반환
        # metadata[:, 0] = label(y), metadata[:, 1] = place(배경), metadata[:, 2] = split
        images, labels, metadata = batch
        images = images.to(device)
        labels = labels.to(device)

        # group 계산: label * 2 + place
        # Group 0: 물새(1) + 물배경(1) → 1*2+1=3 → 재정렬 필요
        # 논문 기준:
        #   Group 0: 물새 + 물배경  (label=1, place=1)
        #   Group 1: 물새 + 땅배경  (label=1, place=0) ★
        #   Group 2: 육지새 + 물배경 (label=0, place=1) ★
        #   Group 3: 육지새 + 땅배경 (label=0, place=0)
        # 공식: group = (1 - label) * 2 + (1 - place)
        #   → label=1,place=1: 0*2+0=0 (Group 0)
        #   → label=1,place=0: 0*2+1=1 (Group 1) ★
        #   → label=0,place=1: 1*2+0=2 (Group 2) ★
        #   → label=0,place=0: 1*2+1=3 (Group 3)
        place        = metadata[:, 0].to(device)
        group_labels = labels.to(device) * 2 + place

        # ── Background Randomization 적용 ─────────────────────────────
        images = randomize_background(
            images,
            center_ratio = config["center_ratio"],
            aug_prob     = config["aug_prob"],
            bg_mode      = config["bg_mode"],
            device       = device,
        )

        # ── Forward / Backward ────────────────────────────────────────
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # ── 통계 집계 ─────────────────────────────────────────────────
        preds          = logits.argmax(dim=1)
        total_loss    += loss.item()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for g in range(config["num_groups"]):
            mask = (group_labels == g)
            if mask.sum() > 0:
                group_correct[g] += (preds[mask] == labels[mask]).sum().item()
                group_total[g]   += mask.sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader)} "
                f"| Loss: {loss.item():.4f} "
                f"| Acc: {total_correct/total_samples*100:.1f}%"
            )

    group_accs = [
        group_correct[g] / group_total[g] if group_total[g] > 0 else 0.0
        for g in range(config["num_groups"])
    ]

    return {
        "avg_loss": total_loss / len(loader),
        "avg_acc":  total_correct / total_samples,
        "group_accs": group_accs,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    num_groups: int,
) -> dict:
    model.eval()

    all_preds  = []
    all_labels = []
    group_correct = np.zeros(num_groups)
    group_total   = np.zeros(num_groups)

    for batch in loader:
        images, labels, metadata = batch
        images = images.to(device)

        # group_labels는 CPU에서 계산 (GPU 연산 불필요)
        place        = metadata[:, 0].long()        # 0=land, 1=water (CPU)
        labels_cpu   = labels.long()                # CPU labels
        group_labels = labels.cpu().long() * 2 + place  # 0~3 (CPU)

        # 평가 시에는 배경 randomization 적용 안 함 (원본 이미지로 평가)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu()
        labels_cpu2 = labels.cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels_cpu2.tolist())

        for g in range(num_groups):
            mask = (group_labels == g)
            if mask.sum() > 0:
                group_correct[g] += (preds[mask] == labels_cpu2[mask]).sum().item()
                group_total[g]   += mask.sum().item()

    group_accs = [
        group_correct[g] / group_total[g] if group_total[g] > 0 else 0.0
        for g in range(num_groups)
    ]

    avg_acc         = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    worst_group_acc = min(group_accs)
    worst_group_idx = group_accs.index(worst_group_acc)

    return {
        "avg_acc":         avg_acc,
        "worst_group_acc": worst_group_acc,
        "worst_group_idx": worst_group_idx,
        "group_accs":      group_accs,
        "group_totals":    group_total.tolist(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  결과 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_result(metrics: dict, split: str, group_names: list):
    print(f"\n{'='*60}")
    print(f"  [{split}] 평가 결과")
    print(f"{'='*60}")
    print(f"  평균 정확도    : {metrics['avg_acc']*100:.1f}%")
    print(f"  Worst-group   : {metrics['worst_group_acc']*100:.1f}%")
    print(f"  -> 최하위: {group_names[metrics['worst_group_idx']]}")
    print(f"\n  그룹별 정확도:")
    for g, (name, acc, n) in enumerate(
        zip(group_names, metrics["group_accs"], metrics["group_totals"])
    ):
        bar    = "█" * int(acc * 20)
        marker = " <- 최하위" if g == metrics["worst_group_idx"] else ""
        print(f"    {name}")
        print(f"      {bar:<20} {acc*100:5.1f}%  (n={int(n)}){marker}")
    print(f"{'='*60}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_bg_randomization(config: dict = CONFIG):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"],    exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Background Randomization 학습 시작")
    print(f"  팀 고유 Contribution")
    print(f"  Device: {device}")
    print(f"{'='*60}")
    print(f"\n  [설정]")
    print(f"  - Epochs       : {config['num_epochs']}")
    print(f"  - Batch size   : {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Center ratio : {config['center_ratio']}  <- 중앙 {int(config['center_ratio']*100)}%를 새로 간주")
    print(f"  - Aug prob     : {config['aug_prob']}  <- {int(config['aug_prob']*100)}% 확률로 배경 교체")
    print(f"  - BG mode      : {config['bg_mode']}")

    # ── 데이터 로딩 (찬민 코드) ──────────────────────────────────────────
    print(f"\n  [데이터] 찬민의 get_waterbirds_loaders 사용...")
    train_loader, val_loader, test_loader = get_waterbirds_loaders(
        batch_size=config["batch_size"]
    )

    # ── 모델 ─────────────────────────────────────────────────────────────
    print(f"  [모델] ResNet-50 pretrained 로딩...")
    model = get_model(config["num_classes"], device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    checkpoint_path      = os.path.join(config["checkpoint_dir"], config["checkpoint_name"])
    best_worst_group_acc = 0.0
    best_epoch           = 0
    history              = {"train": [], "val": []}

    print(f"\n  [학습 시작] Early stopping 기준: Val Worst-group Acc\n")

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        val_metrics = evaluate(model, val_loader, device, config["num_groups"])
        elapsed     = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{config['num_epochs']} ({elapsed:.0f}s) | "
            f"Loss: {train_stats['avg_loss']:.4f} | "
            f"Train Acc: {train_stats['avg_acc']*100:.1f}% | "
            f"Val Avg: {val_metrics['avg_acc']*100:.1f}% | "
            f"Val Worst: {val_metrics['worst_group_acc']*100:.1f}%"
        )

        history["train"].append(train_stats)
        history["val"].append({
            "epoch":           epoch,
            "avg_acc":         val_metrics["avg_acc"],
            "worst_group_acc": val_metrics["worst_group_acc"],
            "group_accs":      val_metrics["group_accs"],
        })

        if val_metrics["worst_group_acc"] > best_worst_group_acc:
            best_worst_group_acc = val_metrics["worst_group_acc"]
            best_epoch           = epoch
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics":     val_metrics,
                    "config":          config,
                },
                checkpoint_path,
            )
            print(f"  [저장] Best 모델! Worst-group: {best_worst_group_acc*100:.1f}%")

    print(f"\n  학습 완료! Best epoch: {best_epoch} | "
          f"Best Val Worst-group: {best_worst_group_acc*100:.1f}%")

    # ── 테스트 평가 ───────────────────────────────────────────────────────
    print(f"\n  [테스트] Best 모델 로드 (epoch {best_epoch})...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device, config["num_groups"])

    # OOD: Group 1 (물새+땅배경), Group 2 (육지새+물배경)
    ood_groups = [1, 2]
    ood_num = sum(test_metrics["group_accs"][g] * test_metrics["group_totals"][g]
                  for g in ood_groups)
    ood_den = sum(test_metrics["group_totals"][g] for g in ood_groups)
    ood_acc = ood_num / ood_den if ood_den > 0 else 0.0

    print_result(test_metrics, "Test", config["group_names"])

    print(f"\n{'='*60}")
    print(f"  최종 결과")
    print(f"{'='*60}")
    print(f"  ID Accuracy        : {test_metrics['avg_acc']*100:.1f}%")
    print(f"  OOD Accuracy       : {ood_acc*100:.1f}%")
    print(f"  Worst-group Acc    : {test_metrics['worst_group_acc']*100:.1f}%")
    print(f"\n  [ERM 대비 비교]")
    print(f"  ERM Worst-group    : 60.0%  (기준선)")
    print(f"  BG Random Worst    : {test_metrics['worst_group_acc']*100:.1f}%  (우리 기여)")
    print(f"{'='*60}")

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    results = {
        "method":          "Background Randomization",
        "team_contribution": True,
        "best_epoch":      best_epoch,
        "config":          config,
        "id_accuracy":     test_metrics["avg_acc"],
        "ood_accuracy":    ood_acc,
        "worst_group_acc": test_metrics["worst_group_acc"],
        "group_accs":      test_metrics["group_accs"],
        "group_names":     config["group_names"],
        "history":         history,
    }

    results_path = os.path.join(config["results_dir"], "bg_randomization_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {results_path}")

    return model, results


if __name__ == "__main__":
    model, results = run_bg_randomization(CONFIG)
