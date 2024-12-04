import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, Resize, Normalize
from wikiart import WikiArtDataset, WikiArtModel

#congig loading
with open("config.json", "r") as f:
    config = json.load(f)

#dataset & dataLoader
transform = Compose([
    Resize((224, 224)),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = WikiArtDataset(config["trainingdir"], transform=transform, device=config["device"])

#address class imbalance by using weighted random sampling 
class_weights = []
for label_name in dataset.classes:
    weight = 1.0 / dataset.label_counts[label_name]
    class_weights.append(weight)

samples_weight = []
for data_entry in dataset:
    image_data, label_data = data_entry
    sample_weight = class_weights[label_data]
    samples_weight.append(sample_weight)

sampler = WeightedRandomSampler(samples_weight, num_samples=len(dataset), replacement=True)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)

#model, optimizer Adam, and loss function
model = WikiArtModel(num_classes=len(dataset.classes), bonusA=True).to(config["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

#train
def train_model(epochs):
    """
    Train the classification model using weighted sampling to handle class imbalance.
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in dataloader:
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Train and save the model
train_model(config["epochs"])
torch.save(model.state_dict(), config["modelsavedfile1"])
