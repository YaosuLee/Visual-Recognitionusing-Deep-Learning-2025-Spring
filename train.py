"""Train a Resnet model for classification."""
import os
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import timm
from tqdm import tqdm
from termcolor import colored
from log_utils.utils import ReDirectSTD
from utils.dataloader import ImageList, ImageList_test
from utils.preprocess import val_transform
from utils.utils import print_model_size, validate, test

class FocalLoss(nn.Module):
    """
    Define the focal loss.
    """
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """ FocalLoss """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss *= alpha_weight

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class CombinedLoss(nn.Module):
    """
    Define the combined loss = crossentropy + focal loss.
    """
    def __init__(self, alpha=None, gamma=2, lambda_focal=0.5):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.lambda_focal = lambda_focal

    def forward(self, logits, targets):
        """ CombinedLoss """
        ce_loss = self.cross_entropy(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        return (1 - self.lambda_focal) * ce_loss + self.lambda_focal * focal_loss

def modify_classifier(model, num_classes=100):
    """Modify the classifier head of the model to include additional layers."""
    num_features = model.get_classifier().in_features
    # print(num_features)
    new_classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        # nn.Linear(1024, 512),
        # nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    if hasattr(model, "classifier"):
        model.classifier = new_classifier
    elif hasattr(model, "fc"):
        model.fc = new_classifier
    else:
        raise ValueError("Unknown classifier structure in model!")

def main():
    """ argument """
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument("--train_dir", type=str, default="data/train_aug")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="timm/resnetaa101d.sw_in12k_ft_in1k")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--schedule", type=str, default="expo", choices=["expo", "multi", "cosine"])
    parser.add_argument("--save_dir", type=str,
                        default="resnetaa101d_rs256_aug_focal_test")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")

    args = parser.parse_args()
    device = torch.device(args.device)
    save_dir = f"{args.save_dir}_sh-{args.schedule}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    log_dir = os.path.join(save_dir, "logs.txt")
    ReDirectSTD(log_dir, "stdout", True)

    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

    model = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes)
    # modify_classifier(model, args.num_classes)

    data_config = timm.data.resolve_model_data_config(model)
    transforms_train = timm.data.create_transform(**data_config, is_training=True)

    print_model_size(model)
    model.to(device)
    model.train()

    train_dataset = ImageList(args.train_dir,
                              transform_w=transforms_train)
    val_dataset = ImageList(args.val_dir,
                            transform_w=val_transform(args.resize_size, args.crop_size))
    test_dataset = ImageList_test(args.test_dir,
                                  transform=val_transform(args.resize_size, args.crop_size))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.bz,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.bz,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.bz,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    alpha = torch.tensor([0.75] * args.num_classes).to(device)  # Đặt alpha = 1.0 cho tất cả class
    # alpha[[8, 12, 13, 15, 16, 18, 34, 35, 48, 50, 52, 53, 56, 57, 62, 69, 83, 88]] = 1.25
    criterion = CombinedLoss(alpha=alpha, gamma=2, lambda_focal=0.4).to(device)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.schedule == "expo":
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif args.schedule == "multi":
        scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler()
    best_valid_acc = 0.0
    epochs_no_improve = 0

    for epoch in tqdm(range(args.epochs), desc="Training epochs"):
        model.train()
        running_loss = 0.0

        for step, batch_train in enumerate(train_loader):
            train_w = batch_train["img_w"].to(device)
            train_labels = batch_train["target"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(train_w)
                loss = criterion(outputs, train_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if step % 20 == 0:
                tb_writer.add_scalar('Training/Loss', loss.item(), epoch * len(train_loader) + step)

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}] Loss: {avg_loss:.6f}")

        print(f"Validating at epoch {epoch+1}")
        val_acc = validate(model, val_loader, device)
        tb_writer.add_scalar('Validation/Accuracy', val_acc, epoch)

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val accuracy: {best_valid_acc:.5f}")
            epochs_no_improve = 0
            test(model, test_loader, device, output_csv=os.path.join(save_dir, "prediction.csv"))
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= args.patience:
            print(colored(f"Early stopping at epoch {epoch+1}",
                          color="red", force_color=True))
            break

    tb_writer.close()
    print(colored(f"Best Validation Accuracy: {best_valid_acc * 100:.2f}%",
                  color="red", force_color=True))

if __name__ == "__main__":
    main()
