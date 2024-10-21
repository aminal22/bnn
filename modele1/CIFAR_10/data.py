import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import requests  # Import the requests library
from PIL import Image  # Import Image for image processing
from io import BytesIO  # Import BytesIO for handling byte streams
from models.nin import Net

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
lr = 0.01
epochs = 20

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize the model
model = Net().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Training loop
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Loss: {running_loss/(batch_idx+1):.3f}')

# Test loop
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    print(f'Test Accuracy: {100. * correct / total:.3f}%')

# Function to test image from URL
def test_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((32, 32))  # Resize image to fit CIFAR-10 input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
        print(f'Predicted class: {predicted.item()}')

# Main function
if __name__ == "__main__":
    for epoch in range(epochs):
        train(epoch)
        test()
        
    # Example URL (replace this with your own image URL)
    image_url = 'https://www.aquaportail.com/pictures2306/cheval.jpg'
    test_image_from_url(image_url)
