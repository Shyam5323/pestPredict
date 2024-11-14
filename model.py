import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


import kagglehub

path = kagglehub.dataset_download("vencerlanz09/agricultural-pests-image-dataset")

print("Path to dataset files:", path)

class SmallVGG_Model(nn.Module):
    def __init__(self, num_classes=12):
        super(SmallVGG_Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 37 * 37, 256),  
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

batch_size = 32
learning_rate = 0.001
num_epochs = 10
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Classes:", train_dataset.classes)

val_dataset = datasets.ImageFolder(root=path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = SmallVGG_Model(num_classes=12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda")

model = model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()  
        outputs = model(inputs) 
        loss = criterion(outputs, labels)  
        loss.backward() 
        optimizer.step()  

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


torch.save(model.state_dict(), 'model.pth')

print("Model training complete and saved to 'model.pth'")
