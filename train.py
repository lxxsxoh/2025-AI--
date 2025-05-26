import argparse
import os
import random
from tqdm import tqdm
from collections import Counter
from PIL import ImageFile

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.nn.functional import avg_pool2d, normalize

from utils import CosineDecayWithWarmup, get_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------- Argument Parsing --------
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_root", type=str, required=True)
parser.add_argument("--classes", nargs=2, required=True)
parser.add_argument("--ratio", type=str, default="1:1")
parser.add_argument("--contrastive", action="store_true")
parser.add_argument("--oversampling", action="store_true")
parser.add_argument("--save_root", type=str, default="./saved_runs")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--warmup_epochs", type=int, default=0)
parser.add_argument("--contrastive_alpha", type=float, default=0.1)
args = parser.parse_args()

# -------- Seed & Setup --------
seed = random.randint(0, 99999)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Save Directory (원하는 형식으로 복구됨) --------
if args.contrastive and args.oversampling:
    exp_type = "contrastive_oversampling"
elif args.contrastive:
    exp_type = "contrastive"
elif args.oversampling:
    exp_type = "oversampling"
else:
    exp_type = "baseline"

# ✅ 예전 스타일: 1:0.01(boar)
ratio_label = f"{args.ratio}({args.classes[1][:3]})"
base_save_dir = os.path.join(args.save_root, exp_type, ratio_label)
os.makedirs(base_save_dir, exist_ok=True)

run_id = 1
while True:
    save_dir = os.path.join(base_save_dir, f'run_{run_id}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        break
    run_id += 1

logger = get_logger(save_dir)
logger.info(f"Seed: {seed} | Classes: {args.classes} | Ratio: {args.ratio} | Contrastive: {args.contrastive} | Oversampling: {args.oversampling}")
logger.info(f"Save path: {save_dir}")

# -------- Transform --------
transform = transforms.Compose([
    transforms.RandAugment(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4446, 0.4303, 0.3811), std=(0.0381, 0.0355, 0.0347)),
])

# -------- Dataset --------
dataset = ImageFolder(root=args.train_data_root, transform=transform)
class_to_idx = dataset.class_to_idx
cls0_idx = class_to_idx[args.classes[0]]
cls1_idx = class_to_idx[args.classes[1]]

# -------- Subsampling --------
ratio_0, ratio_1 = map(float, args.ratio.split(":"))
target_counts = [int(10000 * ratio_0), int(10000 * ratio_1)]

selected = {cls0_idx: [], cls1_idx: []}
for path, label in dataset.samples:
    if label in selected:
        selected[label].append((path, label))

new_samples = []
for idx, count in zip([cls0_idx, cls1_idx], target_counts):
    new_samples.extend(random.sample(selected[idx], min(len(selected[idx]), count)))

dataset.samples = new_samples
dataset.targets = [label for _, label in new_samples]

# -------- Sampler --------
if args.oversampling:
    counter = Counter(dataset.targets)
    weights = {cls: 1.0 / count for cls, count in counter.items()}
    sample_weights = [weights[label] for label in dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
else:
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# -------- Model --------
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

ce_criterion = nn.CrossEntropyLoss()
contrastive_criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = CosineDecayWithWarmup(optimizer, args.warmup_epochs, args.num_epochs)

def get_features_and_out(x, model, layers=[6, 7]):
    feats = {}
    for i, layer in enumerate(model.children()):
        x = layer(x)
        if i in layers:
            feats[i] = x
        if i == 8:
            x = x.flatten(start_dim=-3)
    return feats, x

# -------- Training --------
for epoch in range(args.num_epochs):
    model.train()
    total_ce_loss, total_contrastive_loss = 0, 0
    correct0, total0 = 0, 0
    correct1, total1 = 0, 0

    for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
        x, y = x.to(device), y.to(device)
        feats, out = get_features_and_out(x, model)
        ce_loss = ce_criterion(out, y)

        contrastive_loss = 0
        if args.contrastive:
            for f in feats.values():
                pooled = avg_pool2d(f, f.size(-1)).flatten(1)
                normed = normalize(pooled, p=2, dim=1)
                sim = torch.matmul(normed, normed.T)
                targets = (y.unsqueeze(0) == y.unsqueeze(1)).float()
                contrastive_loss += contrastive_criterion(sim, targets.to(sim.device))

        loss = ce_loss + args.contrastive_alpha * contrastive_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_ce_loss += ce_loss.item() * x.size(0)
        if isinstance(contrastive_loss, torch.Tensor):
            total_contrastive_loss += contrastive_loss.item() * x.size(0)

        pred = out.argmax(dim=1)
        correct0 += ((pred == 0) & (y == 0)).sum().item()
        total0 += (y == 0).sum().item()
        correct1 += ((pred == 1) & (y == 1)).sum().item()
        total1 += (y == 1).sum().item()

    scheduler.step()

    avg_ce_loss = total_ce_loss / len(train_loader.dataset)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader.dataset)
    total_loss = avg_ce_loss + (args.contrastive_alpha * avg_contrastive_loss if args.contrastive else 0)
    total_acc = (correct0 + correct1) / (total0 + total1 + 1e-8)
    class0_acc = correct0 / (total0 + 1e-8)
    class1_acc = correct1 / (total1 + 1e-8)

    logger.info(f"[Epoch {epoch+1}] Train Acc: {total_acc:.4f}, "
                f"Class0 Acc: {class0_acc:.4f}, Class1 Acc: {class1_acc:.4f}, "
                f"Train CE Loss: {avg_ce_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}, "
                f"Total Loss: {total_loss:.4f}")

# -------- Save Model --------
torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
logger.info(f"Model saved at {os.path.join(save_dir, 'best_model.pth')}")
