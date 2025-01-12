import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gcnn_p4.network import P4GConvNet, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, model_path, test_loader, model_name, test_loader_name):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'{model_name} Test Accuracy on the {test_loader_name} dataset: {accuracy:.2f}%')


# Data transformation and test dataset loader
transform = transforms.Compose([
    transforms.RandomRotation(degrees=(90, 90)), # 90 degrees rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

transform_1 = transforms.Compose([
    transforms.RandomRotation(degrees=90), # random rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset_1 = datasets.MNIST(root='./data', train=False, download=True, transform=transform_1)
test_loader_1 = DataLoader(test_dataset_1, batch_size=64, shuffle=False)

evaluate_model(P4GConvNet(), 'gcnn_mnist.pth', test_loader_1, 'G-CNN', 'rotated mnist')
evaluate_model(CNN(), 'cnn_mnist.pth', test_loader_1, 'CNN', 'rotated mnist')

evaluate_model(P4GConvNet(), 'gcnn_mnist.pth', test_loader, 'G-CNN', '90 degrees rotated mnist')
evaluate_model(CNN(), 'cnn_mnist.pth', test_loader, 'CNN', '90 degrees rotated mnist')

# G-CNN Test Accuracy on the rotated mnist dataset: 57.98%
# CNN Test Accuracy on the rotated mnist dataset: 53.97%

#G-CNN Test Accuracy on the 90 degrees rotated mnist dataset: 86.34%
#CNN Test Accuracy on the 90 degrees rotated mnist dataset: 18.26%