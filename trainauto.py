import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from wikiart import WikiArtDataset, Autoencoder
from torchvision.transforms import Compose, Resize, Normalize
import json


with open("config.json", "r") as f:
    config = json.load(f)

transform = Compose([
    Resize((224, 224)),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = WikiArtDataset(config["trainingdir"], transform=transform, device=config["device"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

#autoencoder
autoencoder = Autoencoder().to(config["device"])
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

#train
def train_autoencoder(epochs):
    """
    Train the autoencoder to minimize reconstruction loss.
    """
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0

        for images, labels in dataloader:
            images = images.to(config["device"])
            latent_representation, reconstructed_images = autoencoder(images)
            loss = criterion(reconstructed_images, images)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

#training and savng the autoencoder
train_autoencoder(config["epochs"])
torch.save(autoencoder.state_dict(), config["modelsavedfile2"])

#extracting encodings
autoencoder.eval()
encodings = []
for images, labels in dataloader:
    with torch.no_grad():
        images = images.to(config["device"])
        latent_representation, reconstructed_images = autoencoder(images)
        encodings.append(latent_representation.cpu().view(latent_representation.size(0), -1))

encodings = torch.cat(encodings, dim=0).numpy()

#clustering and PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(encodings)

clustering = AgglomerativeClustering(n_clusters=len(dataset.classes))
clusters = clustering.fit_predict(reduced_data)

# Visualization
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap="tab10", alpha=0.7)
plt.colorbar(label="cluster")
plt.title("PCA with Agglomerative Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
