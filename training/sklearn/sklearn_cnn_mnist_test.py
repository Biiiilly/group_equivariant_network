import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from gcnn_p4.network import P4GConvNet, CNN


def main():
    digits = load_digits()
    X = digits.images  # shape = (n_samples, 8, 8)
    y = digits.target  # shape = (n_samples, )

    X = X / 16.0
    
    X = torch.FloatTensor(X).unsqueeze(1)  # (n_samples, 1, 8, 8)
    y = torch.LongTensor(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def upscale_to_mnist_size(batch_x):
        return F.interpolate(batch_x, size=(28, 28), mode='bilinear', align_corners=False)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    model = CNN()  # or CNN() if you want to train the CNN version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 64

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Upscale to 28x28
            batch_x = upscale_to_mnist_size(batch_x)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.nll_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x = upscale_to_mnist_size(batch_x)

                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "training/sklearn/sklearn_cnn_mnist_model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    main()
