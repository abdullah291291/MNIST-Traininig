import torch
import torch.nn as nn
# import cv2
import model
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

model = model.CNN_Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

### define the hyper-parameter
epochs = 2
batch_size = 32

### preparing the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

print('Starting training...')

# Train the model
for epoch in range(epochs):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print('Finished Training')


### Save the model after training
torch.save(model.state_dict(), 'model.pth')