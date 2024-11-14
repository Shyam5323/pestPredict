import os
from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO
import torch.nn as nn

app = Flask(__name__)

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

model = SmallVGG_Model(num_classes=12)
model.load_state_dict(torch.load('model.pth'))
model.eval()  

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = [
    'Ants', 'Bees', 'Beetles', 'Caterpillars', 'Earthworms',
    'Earwigs', 'Grasshoppers', 'Moths', 'Slugs', 'Snails',
    'Wasps', 'Weevils'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(BytesIO(file.read()))

            img = transform(img).unsqueeze(0) 

            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]

            return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
