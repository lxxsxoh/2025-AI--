import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------- 모델 평가 함수 ----------------------
def evaluate_model(model_path, test_loader, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct_total = 0
    correct_0, total_0 = 0, 0
    correct_1, total_1 = 0, 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)

            correct_total += (preds == labels).sum().item()
            correct_0 += ((preds == 0) & (labels == 0)).sum().item()
            total_0 += (labels == 0).sum().item()
            correct_1 += ((preds == 1) & (labels == 1)).sum().item()
            total_1 += (labels == 1).sum().item()

    overall_acc = correct_total / len(test_loader.dataset)
    acc_0 = correct_0 / (total_0 + 1e-8)
    acc_1 = correct_1 / (total_1 + 1e-8)

    return overall_acc, acc_0, acc_1

# ---------------------- 메인 실행 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=str, required=True, help="e.g., ./trained_model/")
    parser.add_argument("--test_data_root", type=str, required=True, help="e.g., ./dataset/valid/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ---------------------- 데이터셋 로딩 ----------------------
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4446, 0.4303, 0.3811), std=(0.0381, 0.0355, 0.0347)),
    ])
    test_dataset = ImageFolder(root=args.test_data_root, transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---------------------- 폴더 순회 ----------------------
    for exp_type in os.listdir(args.model_root):
        exp_type = exp_type.strip()
        exp_path = os.path.join(args.model_root, exp_type)
        if not os.path.isdir(exp_path):
            continue

        pth_files = [f for f in sorted(os.listdir(exp_path)) if f.endswith(".pth")]
        print(f"\n Experiment Type: {exp_type} ({len(pth_files)} models)")

        for filename in tqdm(pth_files, desc=f"[{exp_type}]"):
            model_path = os.path.join(exp_path, filename)
            overall, acc0, acc1 = evaluate_model(model_path, test_loader, device)

            print(f" {filename}")
            print(f"  ▶ Overall Accuracy : {overall:.4f}")
            print(f"  ▶ Water-deer Accuracy : {acc0:.4f}")
            print(f"  ▶ Roe-deer Accuracy : {acc1:.4f}")


''' python3 /home/nfsyang/deep-learning-demo/newline/test.py \
  --model_root /home/nfsyang/deep-learning-demo/newline/trained_model/ \
  --test_data_root /home/nfsyang/deep-learning-demo/1-waterdeer-vs-roedeer-dataset/valid '''
