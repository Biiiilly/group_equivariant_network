import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gcnn_p4.network import P4GConvNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gcnn = P4GConvNet().to(device)
gcnn.load_state_dict(torch.load('gcnn_mnist.pth'))
gcnn.eval()

transform = transforms.Compose([
    transforms.RandomRotation(degrees=(90, 90)), # Random rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = gcnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy on the rotated mnist dataset: {accuracy:.2f}%')

# Test Accuracy on the rotated mnist dataset: 53.66%
