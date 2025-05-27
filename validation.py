import argparse
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from tqdm import tqdm
from torch.nn.functional import cross_entropy

def find_best_model_paths(root_dir):
    best_models = {}

    for ratio_cls in os.listdir(root_dir):
        ratio_cls_path = os.path.join(root_dir, ratio_cls)
        if not os.path.isdir(ratio_cls_path):
            continue

        best_acc = -1.0
        best_model_path = None

        for run_name in os.listdir(ratio_cls_path):
            run_path = os.path.join(ratio_cls_path, run_name)
            log_path = os.path.join(run_path, "train.log")
            model_path = os.path.join(run_path, "best_model.pth")

            if not os.path.isfile(log_path) or not os.path.isfile(model_path):
                continue

            try:
                with open(log_path, "r") as f:
                    for line in f:
                        match = re.search(r"Train Acc:\s*([0-9.]+)", line)
                        if match:
                            acc = float(match.group(1))
                            if acc > best_acc:
                                best_acc = acc
                                best_model_path = model_path
            except:
                continue

        if best_model_path:
            best_models[ratio_cls] = best_model_path

    return best_models

def evaluate_model(model_path, test_data_root, classes, batch_size=64, log_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4446, 0.4303, 0.3811), std=(0.0381, 0.0355, 0.0347)),
    ])

    dataset = ImageFolder(root=test_data_root, transform=transform)

    #  label remap to force class0=classes[0], class1=classes[1]
    class_to_idx = dataset.class_to_idx
    idx_to_fixed = {class_to_idx[classes[0]]: 0, class_to_idx[classes[1]]: 1}
    def remap(y):
        return torch.tensor([idx_to_fixed[label.item()] for label in y], device=y.device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_loss = 0
    correct = 0
    correct0, total0 = 0, 0
    correct1, total1 = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[Eval] {os.path.basename(os.path.dirname(model_path))}"):
            x, y = x.to(device), remap(y).to(device)

            out = model(x)
            loss = cross_entropy(out, y)
            total_loss += loss.item() * x.size(0)

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            correct0 += ((pred == 0) & (y == 0)).sum().item()
            total0 += (y == 0).sum().item()
            correct1 += ((pred == 1) & (y == 1)).sum().item()
            total1 += (y == 1).sum().item()

    total_acc = correct / len(dataset)
    class0_acc = correct0 / (total0 + 1e-8)
    class1_acc = correct1 / (total1 + 1e-8)
    avg_loss = total_loss / len(dataset)

    # Print
    print(f"\n {os.path.dirname(model_path)}")
    print(f"  ▶ Total Acc      : {total_acc:.4f}")
    print(f"  ▶ water deer ({classes[0]}) Acc : {class0_acc:.4f}")
    print(f"  ▶ roe deer ({classes[1]}) Acc : {class1_acc:.4f}")
    print(f"  ▶ Avg CE Loss    : {avg_loss:.4f}\n")

    # Save to result_log.txt
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n {os.path.dirname(model_path)}\n")
            f.write(f"  ▶ Total Acc      : {total_acc:.4f}\n")
            f.write(f"  ▶ water deer ({classes[0]}) Acc : {class0_acc:.4f}\n")
            f.write(f"  ▶ roe deer ({classes[1]}) Acc : {class1_acc:.4f}\n")
            f.write(f"  ▶ Avg CE Loss    : {avg_loss:.4f}\n")

# -------- Main --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=str, required=True)
    parser.add_argument("--test_data_root", type=str, required=True)
    parser.add_argument("--classes", nargs=2, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    best_models = find_best_model_paths(args.model_root)

    if not best_models:
        print("best_model.pth 를 찾을 수 없습니다.")
        exit(1)

    # 자동 결과 저장 파일 경로 지정 (model_root 하위에 저장)
    result_log_path = os.path.join(args.model_root, "result_log.txt")
    with open(result_log_path, "w") as f:
        f.write(f"[Evaluation Summary]\nClasses: {args.classes}\n\n")

    for ratio_cls, model_path in best_models.items():
        evaluate_model(
            model_path=model_path,
            test_data_root=args.test_data_root,
            classes=args.classes,
            batch_size=args.batch_size,
            log_file=result_log_path
        )
