import os
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import timm
import matplotlib.colors as mcolors
import torch.nn as nn
from utils.dataloader import ImageList
from utils.preprocess import val_transform

def modify_classifier(model, num_classes=100):
    """Modify the classifier head of the model to include additional layers."""
    num_features = model.get_classifier().in_features
    # print(num_features)
    new_classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        # nn.Linear(1024, 512),
        # nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

    # Assign the new classifier to the model
    if hasattr(model, "classifier"):
        model.classifier = new_classifier
    elif hasattr(model, "fc"):  # Some models use 'fc' instead of 'classifier'
        model.fc = new_classifier
    else:
        raise ValueError("Unknown classifier structure in model!")


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png", normalize=True):
    """Generate and save a confusion matrix plot with improvements for large class numbers."""
    cm = confusion_matrix(y_true, y_pred)
    
    # if normalize:
    #     cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", norm=mcolors.LogNorm())

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Normalized Confusion Matrix", fontsize=14)

    plt.xticks(fontsize=8, rotation=90)  # Rotate X labels
    plt.yticks(fontsize=8)  

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def validate(model, dataloader, device):
    """Runs validation and returns true & predicted labels."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["img_w"].to(device)
            labels = batch["target"].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            # print(labels)
            # print(preds)
            # if labels != preds:
            #     print(f'labels: {labels}, pred: {preds}')

    acc = accuracy_score(y_true, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    incorrect = np.array(y_true) != np.array(y_pred)
    missed_label = np.array(y_true)[incorrect]
    missed_pred = np.array(y_pred)[incorrect]
    unique_labels, counts = np.unique(missed_label, return_counts=True)
    print("Missed class counts:")
    for cls, count in zip(unique_labels, counts):
        pred_values = missed_pred[missed_label == cls]  # Lấy tất cả dự đoán sai cho lớp `cls`
        unique_preds, pred_counts = np.unique(pred_values, return_counts=True)  # Đếm từng nhãn sai

        pred_info = ", ".join([f"{p} ({c}x)" for p, c in zip(unique_preds, pred_counts)])
        print(f"Class {cls}: {count} samples misclassified → Predicted as: {pred_info}")

        
    return y_true, y_pred

def main():
    parser = argparse.ArgumentParser(description="Validation and Confusion Matrix Generation")
    parser.add_argument("--test_dir", type=str, default="data/val")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="timm/resnetaa101d.sw_in12k_ft_in1k")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--model_path", type=str, default='/mnt/HDD1/tuong/LamLe/selected/LL/resnetaa101d_rs256_aug_focal0.4_g0.75_class1.25_sh-expo/best_model.pth')
    parser.add_argument("--device", default="cuda:1")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load model
    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    # modify_classifier(model, args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Prepare test dataset
    test_dataset = ImageList(args.test_dir, transform_w=val_transform(args.resize_size, args.crop_size))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Validate & get predictions
    y_true, y_pred = validate(model, test_loader, device)

    # Save normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png", normalize=True)
    print("Normalized confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()
