# test_sklearn_rotation.py

"""
Evaluate P4GConvNet and plain CNN models (trained on the sklearn‑digits dataset)
under two challenging settings:
  1. every test image rotated exactly 90°
  2. every test image rotated by a random angle in [‑90°, +90°]

The script mirrors the structure of your original `test.py`, but swaps in the
sklearn‑digits split and the checkpoints you saved in
`training/sklearn/`.  Make sure those `.pth` files are present before
running.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from gcnn_p4.network import P4GConvNet, CNN

# -----------------------------------------------------------------------------
# Helper dataset wrapper
# -----------------------------------------------------------------------------

class DigitsDataset(Dataset):
    """Wraps (N,1,28,28) tensors and applies an optional transform per sample."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]

# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_test_loader(rotation_transform, batch_size: int = 64):
    """Return DataLoader over the *test* split with the requested rotation."""

    # Load & normalise digits to [0,1]
    digits = load_digits()
    X = torch.FloatTensor(digits.images / 16.0).unsqueeze(1)  # (N,1,8,8)
    y = torch.LongTensor(digits.target)

    # Re‑produce the same 80/20 split used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Upscale from 8×8 → 28×28 to fit the network (which expects MNIST‑sized inputs)
    X_test = F.interpolate(X_test, size=(28, 28), mode="bilinear", align_corners=False)

    dataset = DigitsDataset(X_test, y_test, transform=rotation_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------

def evaluate(model, checkpoint_path: str, loader: DataLoader, *, model_name: str, set_name: str):
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"{model_name:>4s} | {set_name:<28s}: {acc:6.2f}%")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Fixed 90° rotation (clockwise)
    transform_90 = transforms.RandomRotation(degrees=(90, 90))

    # Random rotation in [‑90°, +90°]
    transform_rand = transforms.RandomRotation(degrees=90)

    loader_90 = make_test_loader(transform_90)
    loader_rand = make_test_loader(transform_rand)

    # ---------- G‑CNN ----------
    evaluate(
        P4GConvNet(),
        "training/sklearn/sklearn_gcnn_mnist_model.pth",
        loader_90,
        model_name="G‑CNN",
        set_name="90° rotated digits",
    )
    evaluate(
        P4GConvNet(),
        "training/sklearn/sklearn_gcnn_mnist_model.pth",
        loader_rand,
        model_name="G‑CNN",
        set_name="randomly rotated digits",
    )

    # ---------- Plain CNN ----------
    evaluate(
        CNN(),
        "training/sklearn/sklearn_cnn_mnist_model.pth",
        loader_90,
        model_name="CNN",
        set_name="90° rotated digits",
    )
    evaluate(
        CNN(),
        "training/sklearn/sklearn_cnn_mnist_model.pth",
        loader_rand,
        model_name="CNN",
        set_name="randomly rotated digits",
    )
