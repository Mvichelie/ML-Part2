import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from wikiart import WikiArtDataset, Autoencoder
from torchvision.transforms import Compose, Resize, Normalize
import json


with open("config.json", "r") as f:
    config = json.load(f)

#transforming for resizing and normalizing the images
transform = Compose([
    Resize((224, 224)),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = WikiArtDataset(config["testingdir"], transform=transform, device=config["device"])
dataloader = DataLoader(dataset, batch_size=1) 

#loading the trained autoencoder(from wikiart.py script)
autoencoder = Autoencoder().to(config["device"])
autoencoder.load_state_dict(torch.load(config["modelsavedfile2"]))
#eval mode for autoencoder
autoencoder.eval()  
#creating lists to store encodings and images
encodings = []
original_images = []
reconstructed_images = []

#exxtract latent representations and reconstructed images
with torch.no_grad(): 
     #disable gradient calculation for evaluation
    for images, labels in dataloader:
        images = images.to(config["device"])
        latent_representation, reconstructed = autoencoder(images)
        encodings.append(latent_representation.cpu().view(latent_representation.size(0), -1))
        original_images.append(images.cpu())
        reconstructed_images.append(reconstructed.cpu())

#the latent encodings converted to a numpy array for clustering
encodings = torch.cat(encodings, dim=0).numpy()

#reducing dimensionality with PCA for visualization (2 components)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(encodings)

#applying agglomerative clustering on the reduced data
clustering = AgglomerativeClustering(n_clusters=len(dataset.classes))
clusters = clustering.fit_predict(reduced_data)

def visualize_clusters(save_path="clusters.png"):
    """
    Visualize clusters from the latent representations using PCA.
    Save the plot as an image file instead of displaying it.
    """
    plt.scatter(
        reduced_data[:, 0],  #PCA Component 1 (x-axis)
        reduced_data[:, 1],  #PCA Component 2 (y-axis)
        c=clusters,          #cluster labels as colors
        cmap="Set2",         #color map for clusters
        alpha=0.7            #transparency of points
    )
    plt.colorbar(label="Cluster")
    plt.title("PCA with Agglomerative Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(save_path)  
    print(f"Cluster visualization saved as {save_path}")

def visualize_reconstruction(index=0, save_path="reconstruction.png"):
    """
    Visualizing original and reconstructed images for a specific index (can be changed).
    Saving the plot as an image file instead of displaying it.
    """
    plt.figure(figsize=(8, 4))
    
    #OG image
    plt.subplot(1, 2, 1)
    plt.imshow(original_images[index].squeeze().permute(1, 2, 0).numpy())
    plt.title("Original Image")
    plt.axis("off")
    
    #reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_images[index].squeeze().permute(1, 2, 0).numpy())
    plt.title("Reconstructed Image")
    plt.axis("off")
    
    plt.savefig(save_path)  # Save the visualization as an image
    print(f"Reconstruction visualization saved as {save_path}")

#visualization functions called to save the outputs
visualize_clusters()
visualize_reconstruction(index=0)
